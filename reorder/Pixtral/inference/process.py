import time

def perform_ocr_with_examples(client, image_path, few_shot_examples, model_id, image_file_to_base64):
    system_message = (
        "You are an OCR assistant. For every provided image, extract the text in the correct reading order. "
        "Return your answer as a plain text string exactly as demonstrated in the examples."
    )
    messages = [{"role": "system", "content": system_message}]
    # Add few-shot examples to the conversation
    for i in range(0, len(few_shot_examples), 2):
        human_message = few_shot_examples[i]
        assistant_message = few_shot_examples[i+1]
        image_url = human_message["content"][0]["image_url"]["url"]
        ocr_text = assistant_message["content"]
        messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]})
        messages.append({"role": "assistant", "content": ocr_text})

    # Add final user message (current test image)
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_file_to_base64(image_path)}},
            {"type": "text", "text": "Please extract the ordered text from this image following the demonstrated format shown."}
        ]
    })

    # Call the model
    response = client.chat.complete(
        model=model_id,
        messages=messages,
        temperature=0.0,
        max_tokens=8192
    )
    return response.choices[0].message.content.strip()
