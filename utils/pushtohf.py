from huggingface_hub import HfApi, HfFolder, Repository

trained_model_path = "/Users/kevin/gh/labeldetector/runs/detect/train/weights/best.pt"
hf_repo_name = "label-detector-yolo11n"

print("Make sure you are logged in to Hugging Face: `huggingface-cli login`")

api = HfApi()
username = api.whoami()["name"]
repo_url = api.create_repo(repo_id=hf_repo_name, private=False, exist_ok=True)
print(f"Repository URL: {repo_url}")

local_repo_dir = "/Users/kevin/gh/hf_label_detector_repo"  # Path to clone the repo
repo = Repository(local_dir=local_repo_dir, clone_from=f"{username}/{hf_repo_name}")

import shutil
shutil.copy(trained_model_path, f"{local_repo_dir}/best.pt")

readme_path = f"{local_repo_dir}/README.md"
with open(readme_path, "w") as f:
    f.write(f"# Label Detector with YOLOv11n\n\nThis model was trained on a custom shipping label dataset.")

repo.push_to_hub(commit_message="Initial commit: Upload YOLO model")
print(f"Model successfully uploaded to Hugging Face: {repo_url}")