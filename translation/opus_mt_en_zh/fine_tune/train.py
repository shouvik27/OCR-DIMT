import os
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from .config import *
from .dataset import EnZhDataset
from .data_loader import load_and_split

def get_tokenizer_and_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

def start_execution():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokenizer, model = get_tokenizer_and_model(MODEL_PATH)
    train_data, val_data = load_and_split(DATA_JSON, test_size=0.1, seed=SEED)
    print(f"  → #Train examples = {len(train_data)}")
    print(f"  → #Validation examples = {len(val_data)}")

    train_dataset = EnZhDataset(train_data, tokenizer, max_length=MAX_LENGTH)
    val_dataset   = EnZhDataset(val_data,   tokenizer, max_length=MAX_LENGTH)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=100,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        fp16=True,
        gradient_accumulation_steps=2,
        dataloader_pin_memory=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Running final evaluation on validation set…")
    metrics = trainer.evaluate(eval_dataset=val_dataset)
    print(metrics)
    return metrics
