from ultralytics import YOLO as yolo

task = "detect"
mode = "predict"
model = yolo("yolo11n.pt")
source = "2.webp"

# Run prediction and save results
results = model(source, save=True)

# Alternatively, you can also access the results and save manually:
# for r in results:
#     r.save(filename='result.jpg')  # Save to a specific file
