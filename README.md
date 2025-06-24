# OCR-DIMT

**OCR-DIMT** (Optical Character Recognition Based Document Image Machine Translation) is a modular pipeline for **document image reordering** and further document image machine translation task.  

- **Reorder**: Extracting text from document images in the correct reading order based on ocr input.
- **Translation**: Translating extracted text into the target language.
- **Evaluation**: Assessing system outputs for accuracy and quality.

---

## 📚 Project Structure

```
├── evaluation/                              <- Evaluation pipeline (metrics, scripts, coming soon)
│
├── reorder/                                 <- Reading order extraction pipeline
│   ├── LayoutLMv3_T5/                           <- Baseline: LayoutLMv3 and T5-based methods
│   │   ├── fine_tune/
│   │   │   ├── data/                               <- Data loaders and utilities for fine-tuning
│   │   │   │   ├── __init__.py                        <- Makes data a Python package
│   │   │   │   ├── dataset.py                         <- Custom dataset definition
│   │   │   │   ├── ndjson_reader.py                   <- NDJSON reader utility
│   │   │   ├── __init__.py                        <- Makes fine_tune a Python package
│   │   │   ├── collate.py                         <- Data collation and batch functions
│   │   │   ├── config.py                          <- Fine-tuning configuration for baseline
│   │   │   ├── projection.py                      <- Projection/feature engineering utilities
│   │   │   ├── train.py                           <- Fine-tuning script for LayoutLMv3_T5
│   │   ├── inference/
│   │   │   ├── __init__.py                        <- Makes inference a Python package
│   │   │   └── inference.py                       <- Inference script for LayoutLMv3_T5
│   │
│   ├── Llama_4_Maverick/                          <- Llama 4 Maverick (LLM-based) reorder pipeline
│   │   ├── fine_tune/
│   │   │   ├── __init__.py                        <- Makes fine_tune a Python package
│   │   │   └── train.py                           <- Fine-tuning script for Llama 4 Maverick
│   │   ├── inference/
│   │   │   ├── __init__.py                        <- Makes inference a Python package
│   │   │   └── inference.py                       <- Inference script for Llama 4 Maverick
│   │   ├── .env                                  <- Environment variables for this pipeline
│   │   ├── config.py                             <- Pipeline configuration (inference/fine-tune)
│   │   ├── examples.py                           <- Few-shot examples for inference/fine-tune
│   │   ├── image_utils.py                        <- Image loading/base64 utilities
│   │   ├── inference.py                          <- Inference entry point (script)
│   │   ├── ocr_client.py                         <- Model client interface for Llama 4 Maverick
│   │   └── process.py                            <- Processing and batch utilities
│   │
│   └── Pixtral/                                 <- Pixtral (Mistral LLM-based) reorder pipeline
│       ├── fine_tune/
│       │   ├── __init__.py                        <- Makes fine_tune a Python package
│       │   ├── .env                               <- Environment variables for fine-tuning
│       │   ├── config.py                          <- Fine-tuning configuration
│       │   ├── data_prep.py                       <- Data preparation and formatting utilities
│       │   ├── examples.py                        <- Few-shot/training examples for Pixtral
│       │   ├── fine_tune.py                       <- Fine-tuning script for Pixtral
│       │   ├── image_utils.py                     <- Image conversion/base64 utilities
│       │   ├── ocr_client.py                      <- Pixtral model client
│       │   └── train.py                           <- Training orchestration script
│       ├── inference/
│       │   ├── __init__.py                        <- Makes inference a Python package
│       │   ├── config.py                          <- Inference configuration for Pixtral
│       │   ├── examples.py                        <- Few-shot examples for inference
│       │   ├── image_utils.py                     <- Image utilities for inference
│       │   ├── inference.py                       <- Inference entry point for Pixtral
│       │   ├── ocr_client.py                      <- Model client for inference
│       │   └── process.py                         <- Processing and aggregation scripts
│
├── translation/                             <- Translation pipeline (machine translation and post-processing, coming soon)
│
├── main.py                                  <- Main orchestration script for the project
├── pixtral_training_data.jsonl               <- Example training data for Pixtral (JSONL format)
└── README.md                                <- Project documentation

```