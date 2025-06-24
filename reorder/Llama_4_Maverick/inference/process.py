import time
from langchain.schema.messages import HumanMessage

def perform_ocr_with_examples(image_path, client, system_message, few_shot_examples, image_file_to_base64):
    conversation = [system_message] + few_shot_examples
    final_user_message = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": image_file_to_base64(image_path)}},
            {"type": "text", "text": "Please extract the ordered text from this image following the demonstrated format shown."}
        ]
    )
    conversation.append(final_user_message)
    response = client.invoke(conversation)
    return response.content.strip()
