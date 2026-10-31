from modules.risk import RiskDetector
import cv2

# TODO: tune!
WINSIZE=5
STD_STGNN = 0.1
STD_DETNG = 0.1
STGNN_CHECKPOINT = "dataset_bundle/local_data/stgnn_checkpoint.pth"
STGNN_CALIBRATION = "dataset_bundle/local_data/calib.npz"

def main():
    cam = cv2.VideoCapture(0)
    model = RiskDetector("./assets/ppe_50ep.engine", STGNN_CHECKPOINT,
                         WINSIZE, STD_DETNG, STD_STGNN, graph_rad=0.25,
                         calibrator_path=STGNN_CALIBRATION)
    warmup_frames = []
    for _ in range(WINSIZE):
        ok, warmup_frame = cam.read()
        if not ok:
            cam.release()
            raise RuntimeError("Could not read enough frames to warm up the detector")
        warmup_frames.append(warmup_frame)
    model.preload(warmup_frames)
    while True:
        ret, frame = cam.read()
        if not ret:
           break
        score, frame = model(frame, draw=True)
        reason = model.results.get("model_unavailable_reason")
        if score is None:
            label = f"Risk score unavailable: {reason}"
            print(label)
        else:
            label = f"Risk score: {score:.2f}"
            print(label)
        cv2.putText(
            frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("ppe", frame)
        if cv2.waitKey(1) == ord('q'):
            break

