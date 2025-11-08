from ultralytics import YOLO as yolo

task = "detect"
mode = "predict"
model = yolo("yolo11n.pt")
source = "2.webp"


results = model(source, save=True)
