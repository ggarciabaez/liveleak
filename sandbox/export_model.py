from ultralytics import YOLO

model = YOLO('../assets/models/abombinmycar.pt')
model.export(format="engine", device=0, imgsz=640, quantize=16, conf=0.35, iou=0.65, max_det=100)