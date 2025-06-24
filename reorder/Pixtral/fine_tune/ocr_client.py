import os
import dotenv
from mistralai import Mistral

dotenv.load_dotenv()

def get_mistral_client():
    api_key = os.environ.get("MISTRAL_API_KEY")
    return Mistral(api_key=api_key)

def upload_training_file(client, file_path):
    print(f"Uploading training file: {file_path}")
    with open(file_path, "rb") as f:
        uploaded_file = client.files.upload(
            file={
                "file_name": os.path.basename(file_path),
                "content": f.read(),
            }
        )
    print(f"File uploaded successfully. File ID: {uploaded_file.id}")
    return uploaded_file.id
