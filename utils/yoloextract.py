from ultralytics import YOLO
import cv2
import os
from dotenv import load_dotenv

load_dotenv()

model = YOLO(os.getenv('MODEL_PATH'))

def extract_labels(image_path, model):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    file_extension = os.path.splitext(image_path)[1]

    model = YOLO(os.getenv('MODEL_PATH'))
    results = model(image_path)
    image = cv2.imread(image_path)

    boxes = results[0].boxes.xyxy.tolist()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        ultralytics_crop_object = image[int(y1):int(y2), int(x1):int(x2)]
        
        output_name = f"processed_{base_name}_label{i}.png"

        cv2.imwrite(output_name, ultralytics_crop_object)
        print(f"Object saved at: {output_name}")

extract_labels(os.getenv('TEST_LABEL'), os.getenv('MODEL_PATH'))