import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    set_seed,
    DataCollatorWithPadding,
)
from huggingface_hub import login
import torch
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torch.nn import MSELoss
import torch.nn.functional as F
from transformers import get_scheduler
from transformers.optimization import Adafactor
from peft import LoraConfig, get_peft_model
from hf_token import hf_token
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
login(hf_token)

tokenizer = AutoTokenizer.from_pretrained("saved_models_llama/final_model")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def preprocess_function(examples):
    texts = " ".join(examples["func"].split())
    labels = examples["target"]
    model_inputs = tokenizer(texts, 
                             truncation=True, 
                             padding="max_length", 
                             max_length=512)
    
    return {"input_ids": model_inputs["input_ids"],"attention_mask": model_inputs["attention_mask"],"labels": labels}

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    config = AutoConfig.from_pretrained("saved_models_llama/final_model") 
    config.pad_token_id = tokenizer.pad_token_id
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
    "saved_models_llama/final_model",
    ignore_mismatched_sizes=True,
    config=config,
    ).to(device)
    teacher_model.resize_token_embeddings(len(tokenizer))
    teacher_model.eval()

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Specific modules for LLaMA
    )

    student_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
    student_model = AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        device_map="auto",
        config=student_config,
        torch_dtype=torch.float16,
        ignore_mismatched_sizes=True
    ).to(device)

    student_model.config.pad_token_id = tokenizer.pad_token_id
    student_model.resize_token_embeddings(len(tokenizer))
    student_model = get_peft_model(student_model, lora_config)

    train_dataset = "../dataset/valid.jsonl"

    dataset = load_dataset("json", data_files={"train": train_dataset})
    tokenized_dataset = dataset.map(preprocess_function, remove_columns=['project', 'func', 'target', 'commit_id'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    dataloader = DataLoader(tokenized_dataset["train"], batch_size=8, collate_fn=data_collator)
    optimizer = Adafactor(student_model.parameters(), lr=5e-5, relative_step=False)
    num_training_steps = len(dataloader) * 3  # for 3 epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    #loss_fn = MSELoss()

    student_model.train()
    for epoch in range(3):
        for idx,batch in tqdm(enumerate(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits

            loss =  F.kl_div(F.log_softmax(student_logits/10.00), F.softmax(teacher_logits / 10.00), reduction="batchmean") * (10**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        logger.info(f"Epoch {epoch+1}: Loss {loss.item()}")
    student_model = student_model.merge_and_unload()
    student_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="saved_models_llama/student_model")
    args = parser.parse_args()

    set_seed(42)
    main(args)

