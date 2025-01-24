import os
from huggingface_hub import hf_hub_download

def pull_model_from_hf(key):

    repo_id = "kevinbjoseph/label-detector-yolo11n"
    file_name = "best.pt"

    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            use_auth_token=key
        )
        print(f"File '{file_name}' has been downloaded to: {file_path}")
        return file_path

    except Exception as e:
        print(f"An error occurred while downloading the file: {e}")
        raise
