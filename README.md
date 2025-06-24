# OCR-DIMT

**OCR-DIMT** (Optical Character Recognition Based Document Image Machine Translation) is a modular pipeline for **document image reordering** and further document image machine translation task.  

- **Reorder**: Extracting text from document images in the correct reading order based on ocr input.
- **Translation**: Translating extracted text into the target language.
- **Evaluation**: Assessing system outputs for accuracy and quality.

---

## 📚 Project Structure

```

├── evaluation/                       <- Evaluation pipeline (metrics, comparison, and quality assessment) [details coming soon]
│
├── reorder/                          <- Reading order extraction pipeline for document images
│   ├── LayoutLMv3_T5/                    <- Baseline: LayoutLMv3 and T5-based scripts
│   │   ├── __init__.py                     <- Makes this directory a Python package
│   │   ├── collate.py                      <- Data collation utilities
│   │   ├── config.py                       <- Configuration for the baseline
│   │   ├── projection.py                   <- Projection and feature mapping scripts
│   │   ├── train.py                        <- Training script for LayoutLMv3+T5
│   │   └── inference.py                    <- Inference script for baseline
│   │
│   ├── Llama_4_Maverick/                   <- Llama 4 Maverick pipeline (LLM-based reading order extraction)
│   │   ├── fine_tune/                          <- Scripts and modules for fine-tuning with Llama 4 Maverick
│   │   │   ├── __init__.py                       <- Makes this directory a Python package
│   │   │   ├── config.py                          <- Fine-tune pipeline configuration
│   │   │   ├── examples.py                        <- Few-shot and training examples
│   │   │   ├── ocr_client.py                      <- Llama 4 Maverick model client and interface
│   │   │   ├── process.py                         <- Data and batch processing utilities
│   │   │   └── train.py                           <- Fine-tuning script
│   │   └── inference/                             <- Inference modules for Llama 4 Maverick
│   │       ├── __init__.py                        <- Makes this directory a Python package
│   │       ├── config.py                          <- Inference pipeline configuration
│   │       ├── examples.py                        <- Few-shot examples for inference
│   │       ├── inference.py                       <- Inference script
│   │       ├── ocr_client.py                      <- Model client for inference
│   │       └── process.py                         <- Processing and result aggregation scripts
│   │
│   └── Pixtral/                              <- Pixtral (Mistral) LLM-based reorder pipeline
│       ├── fine_tune/                            <- Fine-tuning utilities and scripts for Pixtral
│       │   ├── __init__.py                           <- Makes this directory a Python package
│       │   ├── config.py                              <- Fine-tuning configuration for Pixtral
│       │   ├── examples.py                            <- Training/few-shot examples for Pixtral
│       │   ├── image_utils.py                          <- Image conversion and base64 utilities
│       │   ├── ocr_client.py                           <- Pixtral model client
│       │   ├── process.py                              <- Data prep and batch processing
│       │   └── train.py                                <- Fine-tuning script
│       └── inference/                              <- Inference utilities for Pixtral
│           ├── __init__.py                           <- Makes this directory a Python package
│           ├── config.py                              <- Inference configuration for Pixtral
│           ├── examples.py                            <- Few-shot examples for inference
│           ├── image_utils.py                         <- Image conversion and base64 utilities
│           ├── inference.py                           <- Main inference script
│           ├── ocr_client.py                          <- Model client for inference
│           └── process.py                             <- Processing and aggregation scripts
│
├── translation/                      <- Translation pipeline (machine translation and post-processing) [details coming soon]
│
├── main.py                           <- Main project orchestration script
├── pixtral_training_data.jsonl        <- Example Pixtral fine-tuning data (JSONL)
└── README.md                         <- Project documentation

```