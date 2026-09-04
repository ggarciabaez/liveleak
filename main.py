from ultralytics import YOLO
import cv2

cam = cv2.VideoCapture(0)
model = YOLO("./assets/abombinmycar.engine")

while True:
    ret, frame = cam.read()
    results = model(frame)
    for result in results:
        frame = result.plot(img=frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
