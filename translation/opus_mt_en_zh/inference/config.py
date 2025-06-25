# config.py

MODEL_DIR = "./finetuned_opus_en_zh"

REORDERED_VAL_JSON  = "/home/hpc/iwi5/iwi5294h/Abantika/reorder/Pixtral/Output/cleaned_extracted_texts_valdataset_fewshot_pixtrall.json"
REORDERED_TEST_JSON = "/home/hpc/iwi5/iwi5294h/Abantika/reorder/Pixtral/Output/cleaned_extracted_texts_testdataset_fewshot_pixtrall.json"

OUT_TRANSLATED_VAL  = "./Output/Validation_Set/translated_reordered_Pixtral_fewshot_val.json"
OUT_TRANSLATED_TEST = "./Output/Test_Set/translated_reordered_Pixtral_fewshot_test.json"

BATCH_SIZE = 32
MAX_LENGTH = 512
NUM_BEAMS  = 5
