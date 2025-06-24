import time

def create_fine_tuning_job(client, training_file_id, model_name="mistral-ocr-latest", suffix=None):
    print("Creating fine-tuning job...")
    job_params = {
        "model": model_name,
        "training_files": [{"file_id": training_file_id, "weight": 1}],
        "hyperparameters": {
            "training_steps": 100,
            "learning_rate": 1e-5,
            "warmup_fraction": 0.05,
        }
    }
    if suffix:
        job_params["suffix"] = suffix
    job = client.fine_tuning.jobs.create(**job_params)
    print(f"Fine-tuning job created successfully. Job ID: {job.id}")
    return job

def monitor_fine_tuning_job(client, job_id, check_interval=60):
    print(f"Monitoring fine-tuning job: {job_id}")
    while True:
        job = client.fine_tuning.jobs.get(job_id=job_id)
        print(f"Job Status: {job.status}")
        if job.status in ["SUCCESS", "FAILED", "CANCELLED"]:
            print(f"Fine-tuning job completed with status: {job.status}")
            if job.status == "SUCCESS":
                print(f"Fine-tuned model ID: {job.fine_tuned_model}")
            return job
        print(f"Job still running... Checking again in {check_interval} seconds")
        time.sleep(check_interval)

def test_fine_tuned_model(client, model_id, test_image_path, image_file_to_base64):
    print(f"Testing fine-tuned model: {model_id}")
    messages = [
        {
            "role": "system",
            "content": "You are an OCR assistant. For every provided image, extract the text in the correct reading order. Return your answer as a plain text string."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_file_to_base64(test_image_path)
                    }
                },
                {
                    "type": "text",
                    "text": "Please extract the ordered text from this image."
                }
            ]
        }
    ]
    response = client.chat.complete(
        model=model_id,
        messages=messages,
        temperature=0.0,
        max_tokens=8192
    )
    return response.choices[0].message.content.strip()

def list_fine_tuning_jobs(client):
    jobs = client.fine_tuning.jobs.list()
    for job in jobs.data:
        print(f"Job ID: {job.id}, Status: {job.status}, Model: {job.model}")
        if job.fine_tuned_model:
            print(f"  Fine-tuned model: {job.fine_tuned_model}")
