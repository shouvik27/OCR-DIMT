# OCR-DIMT

**OCR-DIMT** (Optical Character Recognition Based Document Image Machine Translation) is a modular pipeline for **document image reordering** and further document image machine translation task.  

- **Reorder**: Extracting text from document images in the correct reading order based on ocr input.
- **Translation**: Translating extracted text into the target language.
- **Evaluation**: Assessing system outputs for accuracy and quality.

---

## 📚 Project Structure

## Project Structure

```
OCR-DIMT/
│
├── README.md                       # Project overview and documentation
│
├── evaluation/                     # Scripts for evaluating model performance
│   ├── reorder_bluescore.py        # Evaluation script for reorder task using BLEU score
│   └── translation_bluescore.py    # Evaluation script for translation task using BLEU score
│
├── reorder/                        # Code and data for the document reordering task
│   ├── main.py                     # Main script for running reorder experiments
│   ├── pixtral_training_data.jsonl # Training data for Pixtral model
│   ├── LayoutLMv3_T5/              # LayoutLMv3 T5 model implementation
│   │   ├── fine_tune/              # Fine-tuning scripts and configs
│   │   │   ├── collate.py          # Data collation utilities for training
│   │   │   ├── config.py           # Configuration for fine-tuning
│   │   │   ├── projection.py       # Projection layer implementation
│   │   │   ├── train.py            # Training script for LayoutLMv3 T5
│   │   │   ├── __init__.py         # Package initializer
│   │   │   └── data/               # Data utilities for fine-tuning
│   │   │       ├── dataset.py      # Dataset loader for training
│   │   │       ├── ndjson_reader.py# NDJSON file reader
│   │   │       ├── __init__.py     # Package initializer
│   │   └── inference/              # Inference scripts for LayoutLMv3 T5
│   │       ├── inference.py        # Inference logic for LayoutLMv3 T5
│   │       ├── __init__.py         # Package initializer
│   │
│   ├── Llama_4_Maverick/           # Llama 4 Maverick model implementation
│   │   ├── fine_tune/              # Fine-tuning scripts
│   │   │   ├── train.py            # Training script for Llama 4 Maverick
│   │   │   ├── __init__.py         # Package initializer
│   │   └── inference/              # Inference scripts and utilities
│   │       ├── .env                # Environment variables for inference
│   │       ├── config.py           # Configuration for inference
│   │       ├── examples.py         # Example usage scripts
│   │       ├── image_utils.py      # Image processing utilities
│   │       ├── inference.py        # Inference logic for Llama 4 Maverick
│   │       ├── ocr_client.py       # OCR client utilities
│   │       ├── process.py          # Processing pipeline for inference
│   │       ├── __init__.py         # Package initializer
│   │
│   ├── Pixtral/                    # Pixtral model implementation
│   │   ├── fine_tune/              # Fine-tuning scripts and data preparation
│   │   │   ├── .env                # Environment variables for fine-tuning
│   │   │   ├── config.py           # Configuration for fine-tuning
│   │   │   ├── data_prep.py        # Data preparation script
│   │   │   ├── fine_tune.py        # Fine-tuning logic for Pixtral
│   │   │   ├── image_utils.py      # Image processing utilities
│   │   │   ├── ocr_client.py       # OCR client utilities
│   │   │   ├── train.py            # Training script for Pixtral
│   │   │   ├── __init__.py         # Package initializer
│   │   └── inference/              # Inference scripts for Pixtral
│   │       ├── config.py           # Configuration for inference
│   │       ├── examples.py         # Example usage scripts
│   │       ├── inference.py        # Inference logic for Pixtral
│   │       ├── process.py          # Processing pipeline for inference
│   │       ├── __init__.py         # Package initializer
│
├── translation/                    # Code for the translation task
│   ├── main.py                     # Main script for translation experiments
│   └── opus_mt_en_zh/              # English to Chinese translation model OPUS MT
│       ├── fine_tune/              # Fine-tuning scripts and configs
│       │   ├── config.py           # Configuration for fine-tuning
│       │   ├── dataset.py          # Dataset loader for training
│       │   ├── data_loader.py      # Data loader utilities
│       │   ├── train.py            # Training script for OPUS MT
│       │   ├── __init__.py         # Package initializer
│       └── inference/              # Inference scripts and utilities
│           ├── config.py           # Configuration for inference
│           ├── inference.py        # Inference logic for OPUS MT
│           ├── io_utils.py         # Input output utilities
│           ├── segment_utils.py    # Text segmentation utilities
│           ├── translation_engine.py# Translation engine implementation
│           ├── __init__.py         # Package initializer