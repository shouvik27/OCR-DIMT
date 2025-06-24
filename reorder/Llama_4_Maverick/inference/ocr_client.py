import os
from langchain_groq import ChatGroq
from langchain.schema.messages import SystemMessage

def get_groq_client():
    return ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct"
    )

def get_system_message():
    return SystemMessage(
        content=(
            "You are an OCR assistant. For every provided image, extract the text in the correct reading order. "
            "Return your answer as a plain text string exactly as demonstrated in the examples."
        )
    )
