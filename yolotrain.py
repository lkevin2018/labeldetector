import cv2
from ultralytics import YOLO
import os
from dotenv import load_dotenv

load_dotenv()

model = YOLO("yolo11n.pt")

model.train(data=os.getenv('DATASET_PATH'), batch=8, multi_scale=True, device='mps')
