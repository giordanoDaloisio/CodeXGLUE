import os
import logging
import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from hf_token import hf_token
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
login(hf_token)

def main():
    # Parametri principali
    model_name = "meta-llama/Llama-3.1-8B"
    train_file = "../dataset/train.jsonl"
    eval_file = "../dataset/valid.jsonl"
    output_dir = "./saved_models_llama"
    num_labels = 2
    seed = 42

    set_seed(seed)

    # Carica dataset (assumendo formato JSONL con campi 'func', 'target')
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

    # Carica modello e applica LoRA
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, torch_dtype="auto", device_map="auto")
    # model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to match tokenizer
    model.config.pad_token_id = tokenizer.pad_token

    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Metriche
    metric = load_metric("accuracy", trust_remote_code=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

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
        metric_for_best_model="accuracy",
        seed=seed,
        label_names=["label"]
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Training & Evaluation
    trainer.train()
    trainer.evaluate()

    # Salva il modello
    trainer.save_model(output_dir)
    logger.info("Modello salvato in %s", output_dir)

if __name__ == "__main__":
    main()