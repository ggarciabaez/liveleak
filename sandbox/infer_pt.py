from ultralytics import YOLO
import cv2

model = YOLO("../assets/ppe_50ep.engine", task="detect")
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    results = model.predict(source=frame, quantize=16)
    print(model.track(source=frame, persist=True))

    for result in results:
        frame = result.plot(img=frame)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

