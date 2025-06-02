import os
import logging
import numpy as np
from datasets import load_dataset
from evaluate import load
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
from hf_token import hf_token
from huggingface_hub import login
import torch
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
login(hf_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    model_name = args.model_name
    train_file = "../dataset/train.jsonl"
    eval_file = "../dataset/valid.jsonl"
    output_dir = args.output_dir
    num_labels = 2
    seed = 42

    set_seed(seed)

    def preprocess(example):
        return {"text": " ".join(example["func"].split()), "label": int(example["target"])}

    raw_datasets = load_dataset("json", data_files={"train": train_file, "validation": eval_file})
    raw_datasets = raw_datasets.map(preprocess)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})


    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, torch_dtype="auto")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to match tokenizer
    model.to(device)

    if "Llama" in model_name:
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Specific modules for LLaMA
        )
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["query", "key", "value", "dense"]  # General modules for other models
        )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Metriche
    metric = load("accuracy", trust_remote_code=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        result = metric.compute(predictions=preds, references=labels)
        logger.info("Evaluation result: %s", result)
        return {"eval_accuracy": result["accuracy"]}

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        load_best_model_at_end=True,
        seed=seed,
        label_names=["labels"],
        metric_for_best_model="eval_accuracy",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    # Training & Evaluation
    trainer.train()
    trainer.evaluate()

    # Salva il modello
    trainer.save_model(output_dir)
    logger.info("Modello salvato in %s", output_dir)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./saved_models_llama", help="Directory to save the model")
    
    args = parser.parse_args()

    main(args)