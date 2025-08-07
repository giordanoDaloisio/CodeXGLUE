from __future__ import absolute_import
import os
import copy
import sys
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from tqdm import tqdm, trange
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    RandomSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    WEIGHTS_NAME,
    get_linear_schedule_with_warmup,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.optimization import Adafactor
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import time
from huggingface_hub import login
from hf_token import hf_token

login(hf_token)

MODEL_CLASSES = {
    "llama": (LlamaConfig, AutoModelForCausalLM, AutoTokenizer),
}

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(
        self,
        idx,
        source,
        target,
    ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if "idx" not in js:
                js["idx"] = idx
            code = " ".join(js["code_tokens"]).replace("\n", " ")
            code = " ".join(code.strip().split())
            nl = " ".join(js["docstring_tokens"]).replace("\n", "")
            nl = " ".join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
        self,
        example_id,
        input_ids,
        attention_mask,
        labels,
    ):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # Formato: "Code: <code> Summary: <summary>"
        if stage == "test":
            input_text = f"Code: {example.source} Summary:"
            target_text = ""
        else:
            input_text = f"Code: {example.source} Summary: {example.target}"
            target_text = example.target

        # Tokenizza l'input completo
        inputs = tokenizer(
            input_text,
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        
        if stage != "test":
            # Per il training, le labels sono gli input_ids shiftati
            labels = input_ids.clone()
            # Maschera le parti di input (solo la summary deve essere predetta)
            source_length = len(tokenizer(f"Code: {example.source} Summary:")["input_ids"])
            labels[:source_length] = -100  # Ignora la loss sulla parte di input
        else:
            labels = torch.tensor([-100] * len(input_ids))

        if example_index < 5 and stage == "train":
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example.idx))
            logger.info("input_text: {}".format(input_text))
            logger.info("input_ids: {}".format(input_ids))

        features.append(
            InputFeatures(
                example_index,
                input_ids,
                attention_mask,
                labels,
            )
        )
    return features


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def print_model_size(model, args):
    torch.save(model.state_dict(), f"tmp_{args.job_id}.p")
    print("Size (MB): " + str(os.path.getsize(f"tmp_{args.job_id}.p") / 1e6))
    os.remove(f"tmp_{args.job_id}.p")

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--model_type",
        default="llama",
        type=str,
        help="Model type: llama",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="meta-llama/Llama-3.1-8B",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--load_model_path",
        default=None,
        type=str,
        help="Path to trained model: Should contain the PEFT adapter files",
    )
    
    # LoRA parameters
    parser.add_argument("--lora_r", default=16, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=32, type=int, help="LoRA alpha")
    parser.add_argument("--lora_dropout", default=0.1, type=float, help="LoRA dropout")
    
    ## Other parameters
    parser.add_argument(
        "--train_filename",
        default=None,
        type=str,
        help="The train filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--dev_filename",
        default=None,
        type=str,
        help="The dev filename. Should contain the .jsonl files for this task.",
    )
    parser.add_argument(
        "--test_filename",
        default=None,
        type=str,
        help="The test filename. Should contain the .jsonl files for this task.",
    )

    parser.add_argument(
        "--max_source_length",
        default=512,
        type=int,
        help="The maximum total source sequence length after tokenization.",
    )
    parser.add_argument(
        "--max_target_length",
        default=128,
        type=int,
        help="The maximum total target sequence length after tokenization.",
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the test set."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    
    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-4,
        type=float,
        help="The initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument("--job_id", help="job id to put on logfile")

    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
        
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )
    args.device = device
    set_seed(args.seed)
    
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    # Carica tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    

    model = model_class.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto" if args.n_gpu > 1 else None
    )
        
    # Aggiungi padding token se non presente
    # if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token to eos token for generation compatibility
    # if model.config.pad_token_id is None:
    #     model.config.pad_token_id = tokenizer.pad_token_id
    
        # Carica modello base
        # Ridimensiona embeddings se necessario
        # model.resize_token_embeddings(len(tokenizer))
    
    if args.do_train:
        # Configura LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        
        # Applica LoRA al modello
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Carica checkpoint se specificato
    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        # Don't create a new PeftModel, just load the adapter weights
        model.load_adapter(args.load_model_path, "default")

    model.to(device)

    ########### TRAINING #############
    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args, stage="train"
        )
        
        all_input_ids = torch.stack([f.input_ids for f in train_features])
        all_attention_mask = torch.stack([f.attention_mask for f in train_features])
        all_labels = torch.stack([f.labels for f in train_features])
        
        train_data = TensorDataset(all_input_ids, all_attention_mask, all_labels)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
            
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size // args.gradient_accumulation_steps,
        )

        # Prepare optimizer and schedule
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=False, 
            relative_step=False, 
            warmup_init=False, 
            lr=args.learning_rate)

        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        model.train()
        global_step, best_bleu = 0, 0
        
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            tr_loss = 0
            
            for step, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                tr_loss += loss.item()
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                train_loss = tr_loss / (step + 1)
                bar.set_description("epoch {} loss {:.4f}".format(epoch, train_loss))

            # Save checkpoint after each epoch
            output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save only the LoRA adapter weights during training
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # If you want to save the final merged model, do it only after training is complete
            if epoch == args.num_train_epochs - 1:  # Last epoch
                final_output_dir = os.path.join(args.output_dir, "final_merged_model")
                if not os.path.exists(final_output_dir):
                    os.makedirs(final_output_dir)
                
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(final_output_dir)
                tokenizer.save_pretrained(final_output_dir)

    ######## INFERENCE #############
    if args.do_test:
        
        print_model_size(model, args)

        logfile = f"times_{args.job_id}_{'cuda' if torch.cuda.is_available() else 'cpu'}.csv"
        time_dir = os.path.join(args.output_dir, "times")
        os.makedirs(time_dir, exist_ok=True)
        if os.path.exists(os.path.join(time_dir, logfile)):
            os.remove(os.path.join(time_dir, logfile))

        files = []
        if args.dev_filename is not None:
            files.append(args.dev_filename)
        if args.test_filename is not None:
            files.append(args.test_filename)
        

        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file)
            eval_features = convert_examples_to_features(
                eval_examples, tokenizer, args, stage="test"
            )
            
            all_input_ids = torch.stack([f.input_ids for f in eval_features])
            all_attention_mask = torch.stack([f.attention_mask for f in eval_features])
            
            eval_data = TensorDataset(all_input_ids, all_attention_mask)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
            )
            
            model.eval()
            p = []
            times = []
            ## GPU warm-up
            if torch.cuda.is_available() and not args.no_cuda:
                logger.info("********* GPU-WARM-UP ********")
                # warm up for 5 batches
                for i, batch in enumerate(eval_dataloader):
                    if i < 5:
                        batch = tuple(t.to(device) for t in batch)
                        source_ids, source_mask = batch
                        with torch.no_grad():
                            _ = model.generate(
                                input_ids=source_ids,
                                attention_mask=source_mask,
                            )
                    else:
                        break
            
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask = batch
                
                with torch.no_grad():
                    # Generate usando il modello
                    start_time = time.time()
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_target_length,
                        num_beams=4,
                        early_stopping=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    # Decodifica solo la parte generata (dopo l'input)
                    for i, generated in enumerate(generated_ids):
                        input_length = len(input_ids[i])
                        generated_text = tokenizer.decode(
                            generated[input_length:], 
                            skip_special_tokens=True
                        )
                        p.append(generated_text.strip())
                    times.append(elapsed_time)
                    with open(os.path.join(time_dir, logfile), "a") as f:
                        f.write(str(elapsed_time) + ",")
                    torch.cuda.synchronize()
            
            # Salva risultati
            predictions = []
            output_file = f"test_{idx}.output"
            gold_file = f"test_{idx}.gold"
            
            with open(os.path.join(args.output_dir, output_file), "w") as f, open(
                os.path.join(args.output_dir, gold_file), "w"
            ) as f1:
                for ref, gold in zip(p, eval_examples):
                    predictions.append(str(gold.idx) + "\t" + ref)
                    f.write(str(gold.idx) + "\t" + ref + "\n")
                    f1.write(str(gold.idx) + "\t" + gold.target + "\n")

            # Calcola BLEU score
            (goldMap, predictionMap) = bleu.computeMaps(
                predictions, os.path.join(args.output_dir, gold_file)
            )
            dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
            logger.info("Average inference time: " + str(np.mean(times)))
            logger.info("  " + "*" * 20)



if __name__ == "__main__":
    main()