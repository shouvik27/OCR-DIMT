import json
import os
import torch
from transformers import (
    AutoProcessor,
    LayoutLMv3Model,
    T5ForConditionalGeneration,
    AutoTokenizer
)
from torch.utils.data import Dataset, DataLoader
from .config import Config
from .data import iter_ndjson_in_chunks, OcrInferenceDataset
from .collate import InferenceCollator, t5_tokenizer
from .projection import build_projection

def run_inference(
    test_json=Config.TEST_JSON,
    test_img_dir=Config.TEST_IMG_DIR,
    output_json=None,
    checkpoint_dir=Config.CHECKPOINT_DIR,
    epoch=30,
    device=None,
    batch_size=Config.BATCH_SIZE,
    chunk_size=Config.CHUNK_SIZE,
    max_output_length=Config.MAX_OUTPUT_LENGTH
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if output_json is None:
        output_json = os.path.join(checkpoint_dir, "predictions.json")

    model_ckpt = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    proc_ckpt = os.path.join(checkpoint_dir, f"processor_epoch_{epoch}")

    processor = AutoProcessor.from_pretrained(proc_ckpt, apply_ocr=False)
    layout_model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    projection = build_projection(768, t5_model.config.d_model)

    ckpt = torch.load(model_ckpt, map_location="cpu")
    layout_model.load_state_dict(ckpt['layout_model'])
    t5_model.load_state_dict(ckpt['t5_model'])
    projection.load_state_dict(ckpt['projection'])

    layout_model.to(device).eval()
    t5_model.to(device).eval()
    projection.to(device).eval()

    with open(output_json, "w", encoding="utf-8") as writer:
        writer.write("{\n")
        first_entry = True

        for chunk in iter_ndjson_in_chunks(test_json, chunk_size=chunk_size):
            ds = OcrInferenceDataset(chunk, test_img_dir)
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=InferenceCollator(processor)
            )

            for batch in loader:
                pv = batch["pixel_values"].to(device)
                mask = batch["attention_mask"].to(device)
                bbox = batch["bbox"].to(device)
                img_names = batch["img_names"]
                input_ids = batch["input_ids"].to(device)

                with torch.no_grad(), torch.cuda.amp.autocast():
                    lm_out = layout_model(
                        pixel_values=pv,
                        input_ids=input_ids,
                        attention_mask=mask,
                        bbox=bbox
                    )
                    seq_len = input_ids.size(1)
                    text_feats = lm_out.last_hidden_state[:, :seq_len, :]
                    proj_feats = projection(text_feats)

                    gen_ids = t5_model.generate(
                        inputs_embeds=proj_feats,
                        attention_mask=mask,
                        max_length=max_output_length
                    )

                texts = t5_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

                for img_name, txt in zip(img_names, texts):
                    if not first_entry:
                        writer.write(",\n")
                    first_entry = False
                    writer.write(f"{json.dumps(img_name)}: {json.dumps(txt, ensure_ascii=False)}")

                writer.flush()
                del pv, mask, bbox, lm_out, text_feats, proj_feats, gen_ids, input_ids
                torch.cuda.empty_cache()

            del ds, loader
            torch.cuda.empty_cache()

        writer.write("\n}")
    print(f"Inference complete — results written to {output_json}")

if __name__ == "__main__":
    run_inference()