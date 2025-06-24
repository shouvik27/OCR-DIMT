from .config import *
from .data_prep import prepare_training_data
from .ocr_client import get_mistral_client, upload_training_file
from .fine_tune import create_fine_tuning_job, monitor_fine_tuning_job

def start_execution():
    print("Starting Pixtral 12B fine-tuning process...")
    # Step 1: Prepare training data
    training_file, num_samples = prepare_training_data(
        TRAIN_JSON_PATH,
        TRAIN_IMG_DIR,
        TRAINING_FILE_PATH,
        max_samples=MAX_TRAINING_SAMPLES
    )
    print(f"Training data prepared: {num_samples} samples in {training_file}")
    # Step 2: Upload training file
    client = get_mistral_client()
    training_file_id = upload_training_file(client, training_file)
    # Step 3: Create fine-tuning job
    job = create_fine_tuning_job(
        client,
        training_file_id,
        model_name=MODEL_NAME,
        suffix=SUFFIX
    )
    # Step 4: Monitor the job
    completed_job = monitor_fine_tuning_job(client, job.id)
    # Step 5: Print result
    if completed_job.status == "SUCCESS":
        print(f"\nFine-tuning completed successfully!")
        print(f"Your fine-tuned model ID: {completed_job.fine_tuned_model}")
    else:
        print(f"Fine-tuning failed with status: {completed_job.status}")


