from modules.risk import RiskDetector
import cv2

# TODO: tune!
WINSIZE=5
STD_STGNN = 0.1
STD_DETNG = 0.1

def main():
    cam = cv2.VideoCapture(0)
    model = RiskDetector("./assets/ppe_50ep.engine", "./assets/stgnn.pth",
                         WINSIZE, STD_DETNG, STD_STGNN, graph_rad=250)
    model.preload([cam.read()[1] for _ in range(WINSIZE)])
    while True:
        ret, frame = cam.read()
        if not ret:
           break
        score, frame = model(frame, draw=True)
        cv2.imshow("ppe", frame)
        if cv2.waitKey(1) == ord('q'):
            break

