from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj=r"pytorch_model.bin",  # or pytorch_model.bin
    path_in_repo="pytorch_model.bin",                   # How it will be named on HF
    repo_id="shouvik27/LayoutLMv3_T5",
    repo_type="model"
)
