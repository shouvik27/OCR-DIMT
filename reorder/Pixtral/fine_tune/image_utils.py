import os
import base64

def image_file_to_base64(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        mime_type = "image/jpeg"
    elif ext == '.png':
        mime_type = "image/png"
    else:
        mime_type = "application/octet-stream"
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"
