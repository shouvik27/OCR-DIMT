"""Training script for LayoutLMv3 + T5 model."""

import sys
sys.setrecursionlimit(10000)  # Increase recursion limit to avoid RecursionError

import os
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
import torch.nn.utils as nn_utils
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    LayoutLMv3Model,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)

from .config import TrainConfig, ModelConfig
from .data.ndjson_reader import iter_ndjson_in_chunks
from .data.dataset import OcrReorderDataset
from .collate import CustomCollator
from .projection import build_projection

def start_execution():
    """Main training function."""
    # Initialize configuration
    config = TrainConfig()
    model_config = ModelConfig()
    
    # Setup directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models and processor
    processor = AutoProcessor.from_pretrained(model_config.layoutlm_model_name, apply_ocr=False)
    layout_model = LayoutLMv3Model.from_pretrained(model_config.layoutlm_model_name).to(device)
    t5_model = T5ForConditionalGeneration.from_pretrained(model_config.t5_model_name).to(device)
    projection = build_projection(t5_model).to(device)
    
    # Optimizer
    optimizer = AdamW(
        list(layout_model.parameters()) +
        list(t5_model.parameters()) +
        list(projection.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    steps_per_ep = math.ceil(config.max_samples / config.batch_size)
    total_steps = steps_per_ep * config.num_epochs
    warmup_steps = int(config.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Data collator
    data_collator = CustomCollator(processor)
    global_step = 0
    
    # Training loop
    for epoch in range(1, config.num_epochs + 1):
        print(f"\n===== STARTING EPOCH {epoch}/{config.num_epochs} =====")
        processed = 0
        chunk_idx = 0
        epoch_loss = 0.0
        epoch_samples = 0

        for chunk in iter_ndjson_in_chunks(config.train_json, chunk_size=config.chunk_size):
            if processed >= config.max_samples:
                break
            if processed + len(chunk) > config.max_samples:
                chunk = chunk[: (config.max_samples - processed)]
            processed += len(chunk)
            chunk_idx += 1
            print(f"  --> Chunk {chunk_idx}: {len(chunk)} samples (Total {processed}/{config.max_samples})")

            dataset = OcrReorderDataset(chunk, config.train_img_dir, processor)
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=data_collator
            )

            layout_model.train(); t5_model.train(); projection.train()
            total_loss = 0
            chunk_samples = 0
            bar = tqdm(loader, desc=f"Epoch {epoch} Chunk {chunk_idx}")

            for batch in bar:
                optimizer.zero_grad()
                pv = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                bbox = batch["bbox"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast():
                    # Forward pass
                    layout_out = layout_model(
                        pixel_values=pv,
                        input_ids=input_ids,
                        attention_mask=mask,
                        bbox=bbox
                    )
                    seq_len = input_ids.size(1)
                    text_feats = layout_out.last_hidden_state[:, :seq_len, :]
                    proj_feats = projection(text_feats)
                    outputs = t5_model(
                        inputs_embeds=proj_feats,
                        attention_mask=mask,
                        labels=labels
                    )
                    loss = outputs.loss

                # Backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn_utils.clip_grad_norm_(
                    list(layout_model.parameters()) +
                    list(t5_model.parameters()) +
                    list(projection.parameters()),
                    max_norm=config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Log metrics
                batch_loss = loss.item()
                total_loss += batch_loss
                epoch_loss += batch_loss * len(batch["input_ids"])
                epoch_samples += len(batch["input_ids"])
                chunk_samples += len(batch["input_ids"])
                global_step += 1
                
                writer.add_scalar('Loss/train_batch', batch_loss, global_step)
                writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], global_step)
                
                bar.set_postfix(loss=batch_loss, lr=scheduler.get_last_lr()[0])

            # Log chunk metrics
            avg_chunk_loss = total_loss / len(loader)
            writer.add_scalar('Loss/train_chunk', avg_chunk_loss, global_step)
            print(f"Chunk {chunk_idx} done - Avg Loss: {avg_chunk_loss:.4f}")

            del dataset, loader
            torch.cuda.empty_cache()

        # Log epoch metrics
        avg_epoch_loss = epoch_loss / epoch_samples
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
        writer.add_scalar('Loss/train_epoch_avg', avg_epoch_loss, epoch)
        print(f"Epoch {epoch} complete - Avg Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if epoch % 5 == 0:
            ckpt_path = os.path.join(config.save_dir, f"model_epoch_{epoch}.pth")
            proc_path = os.path.join(config.save_dir, f"processor_epoch_{epoch}")
            print(f"Saving checkpoint for epoch {epoch}")
            torch.save({
                'layout_model': layout_model.state_dict(),
                't5_model': t5_model.state_dict(),
                'projection': projection.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
                'avg_loss': avg_epoch_loss
            }, ckpt_path)
            processor.save_pretrained(proc_path)
            writer.flush()

    writer.close()
    print("\nTraining complete. Models saved every 5 epochs.")
