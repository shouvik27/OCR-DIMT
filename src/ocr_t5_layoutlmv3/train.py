import math
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LayoutLMv3Model,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import torch.nn.utils as nn_utils

from .config import Config
from .data import iter_ndjson_in_chunks, OcrReorderDataset
from .collate import CustomCollator
from .projection import build_projection

def main():
    Config.create_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(Config.CHECKPOINT_DIR, "logs"))

    # Initialize models
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    layout_model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base").to(device)
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    projection = build_projection(768, t5_model.config.d_model).to(device)

    # Optimizer
    optimizer = AdamW(
        list(layout_model.parameters()) +
        list(t5_model.parameters()) +
        list(projection.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    steps_per_ep = math.ceil(Config.MAX_SAMPLES / Config.BATCH_SIZE)
    total_steps = steps_per_ep * Config.NUM_EPOCHS
    warmup_steps = int(Config.WARMUP_PCT * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training setup
    scaler = torch.cuda.amp.GradScaler()
    data_collator = CustomCollator(processor)
    global_step = 0

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\n===== STARTING EPOCH {epoch}/{Config.NUM_EPOCHS} =====")
        processed = 0
        chunk_idx = 0
        epoch_loss = 0.0
        epoch_samples = 0

        for chunk in iter_ndjson_in_chunks(Config.TRAIN_JSON, chunk_size=Config.CHUNK_SIZE):
            if processed >= Config.MAX_SAMPLES:
                break
            if processed + len(chunk) > Config.MAX_SAMPLES:
                chunk = chunk[: (Config.MAX_SAMPLES - processed)]
            processed += len(chunk)
            chunk_idx += 1
            print(f"  --> Chunk {chunk_idx}: {len(chunk)} samples (Total {processed}/{Config.MAX_SAMPLES})")

            dataset = OcrReorderDataset(chunk, Config.TRAIN_IMG_DIR, processor)
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
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

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn_utils.clip_grad_norm_(
                    list(layout_model.parameters()) +
                    list(t5_model.parameters()) +
                    list(projection.parameters()),
                    max_norm=Config.GRADIENT_CLIP
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                batch_loss = loss.item()
                total_loss += batch_loss
                epoch_loss += batch_loss * len(batch["input_ids"])
                epoch_samples += len(batch["input_ids"])
                chunk_samples += len(batch["input_ids"])
                global_step += 1
                
                writer.add_scalar('Loss/train_batch', batch_loss, global_step)
                writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], global_step)
                bar.set_postfix(loss=batch_loss, lr=scheduler.get_last_lr()[0])

            avg_chunk_loss = total_loss / len(loader)
            writer.add_scalar('Loss/train_chunk', avg_chunk_loss, global_step)
            print(f"Chunk {chunk_idx} done - Avg Loss: {avg_chunk_loss:.4f}")

            del dataset, loader
            torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / epoch_samples
        writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
        writer.add_scalar('Loss/train_epoch_avg', avg_epoch_loss, epoch)
        print(f"Epoch {epoch} complete - Avg Loss: {avg_epoch_loss:.4f}")

        if epoch % 5 == 0:
            ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
            proc_path = os.path.join(Config.CHECKPOINT_DIR, f"processor_epoch_{epoch}")
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

if __name__ == "__main__":
    main()