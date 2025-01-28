import os
import cv2
from ultralytics import YOLO
from zebraprint.zbprint import print_label
import logging

logging.basicConfig(
    level=logging.INFO,
    filename='yoloextract.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_labels(image_path, model_path, friendly_name, save_directory, isPrint=False, print_path=None):
    os.makedirs(save_directory, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    file_extension = os.path.splitext(image_path)[1]

    model = YOLO(model_path)

    results = model(image_path)
    image = cv2.imread(image_path)

    boxes = results[0].boxes.xyxy.tolist()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        ultralytics_crop_object = image[int(y1):int(y2), int(x1):int(x2)]
        
        height, width, _ = ultralytics_crop_object.shape
        if width > height:
            ultralytics_crop_object = cv2.rotate(ultralytics_crop_object, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        output_name = f"processed_{friendly_name}_label{i}.png"
        output_path = os.path.join(save_directory, output_name)

        cv2.imwrite(output_path, ultralytics_crop_object)
        if isPrint:
            logging.info(f'Attempting to print {output_name}')
            print_label(print_path, output_path)
            print(f"Object saved at: {output_path}")
        else:
            logging.info(f"Object: {output_name} saved at: {output_path}")
            print(f"Object saved at: {output_path}")