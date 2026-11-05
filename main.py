import cv2

from modules.risk import RiskDetector

# TODO: tune!
WINSIZE=8
STD_STGNN = 0.1
STD_DETNG = 0.1

def main():
    cam = cv2.VideoCapture("./data/trainvid_0.mp4")
    # cam = cv2.VideoCapture(0)
    model = RiskDetector("./assets/ppe_50ep.engine", "./assets/v2/stgnn.pth",
                         "./assets/v2/calib.npz", WINSIZE, STD_DETNG, STD_STGNN, graph_rad=0.6,
                         video_width=cam.get(cv2.CAP_PROP_FRAME_WIDTH) or 640,
                         video_height=cam.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480,
                         )
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

