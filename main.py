from modules.risk import RiskDetector
import cv2

# TODO: tune!
WINSIZE=5
STD_STGNN = 0.1
STD_DETNG = 0.1

def main():
    cam = cv2.VideoCapture(0)
    model = RiskDetector("./assets/ppe_50ep.engine", "./assets/stgnn.pth",
                         "./assets/calib.npz", WINSIZE, STD_DETNG, STD_STGNN, graph_rad=250)
    model.preload_direct(cam)
    while True:
        ret, frame = cam.read()
        if not ret:
           break
        score, frame = model(frame, draw=True)
        print("Score:", score)
        if score is None:
            cv2.putText(frame, "No detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Score: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("ppe", frame)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()

