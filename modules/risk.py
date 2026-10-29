from modules.Detector import Detector
from modules.stgnn import STGNN, MotionAnomalyCalibrator, torch, np
from modules.detng import DetNG, get_rules
from modules.kfilter import KalmanFilter
from collections import deque
from tqdm import tqdm
import cv2


class RiskDetector:
    def __init__(self, yolo_path, stgnn_path, calib_path,
                 winsize=5, std_detng=0.1, std_stgnn=0.1,
                 graph_rad=250, ruleset=None):
        # The pipeline. quite simple, really.
        self.detector = Detector(yolo_path)
        self.detng = DetNG().compile(get_rules() if ruleset is None else ruleset)
        self.model = STGNN(graph_rad)
        self.calibrator = MotionAnomalyCalibrator()

        self.calibrator.load(calib_path)
        self.model.load_weights(stgnn_path)

        self.kf = KalmanFilter()
        self.results = {}

        if torch.cuda.is_available():
            self.model.to('cuda')

        self.qvec = deque(maxlen=winsize)
        self.sigdet = std_detng
        self.sigstg = std_stgnn
        self.GRID = 32  # i don't even want to begin explaining this.

    def merge_frame(self, vectors):
        """get_vectors() output: list of 4 (n_c, 13) arrays -> one (n, 13) array.
        Category is already in cols 1-4 (one-hot), so no category axis is needed."""
        f = np.concatenate(vectors, axis=0)
        cone = f[:, 0] == -1
        if cone.any():
            gx = (f[cone, 5] // self.GRID).astype(int)
            gy = (f[cone, 6] // self.GRID).astype(int)
            f[cone, 0] = -(2 + gx * 10000 + gy)  # always <= -2, never collides with -1 pad or real ids
        return f

    def build_window(self, frames):  # list of (n_t, 13)
        # TODO: this overwrites cones because they share the same ID.
        ids = sorted({int(i) for f in frames for i in f[:, 0]})
        slot = {i: k for k, i in enumerate(ids)}
        T, N = len(frames), len(ids)
        feats = np.zeros((T, N, 13), np.float32)
        feats[..., 0] = -1
        valid = np.zeros((T, N), bool)
        for t, f in enumerate(frames):
            s = [slot[int(i)] for i in f[:, 0]]
            feats[t, s] = f
            valid[t, s] = True
        # valid is a (T, N) mask that indicates if vector N was present in frame T.
        return feats, valid

    def preload(self, frames):
        for frame in tqdm(frames):
            self._prime_run(frame)

    def preload_direct(self, cam):
        while len(self.qvec) < self.qvec.maxlen:
            ret, frame = cam.read()
            if not ret:
                raise ValueError("Failed to read frame from camera")
            self._prime_run(frame)

    def _prime_run(self, frame):
        vecs = self.detector.get_vectors(frame)  # TODO: there might be a bug with the IOU code.
        # Sometimes it detects me (and only me) as 2 vectors.
        vecs = self.merge_frame(vecs)
        self.qvec.append(vecs)
        nvecs, mask = self.build_window(self.qvec)
        score, *_ = self.detng.execute(nvecs, mask)
        if score is None:
            self.qvec.pop()
            return
        self.kf.fuse(0.0, score, self.sigstg, self.sigdet)

    def __call__(self, frame, draw=False):
        dets = self.detector(frame)
        rvecs = self.detector.get_vectors(dets)
        vecs = self.merge_frame(rvecs)
        if vecs.shape[0] == 0:
            print("WARNING: no detections")
            return None, frame
        self.qvec.append(vecs)
        fvecs, mask = self.build_window(self.qvec)
        ngscore, violations, ruleout = self.detng.execute(fvecs, mask)

        with torch.inference_mode():
            modelscore = self.model.compute_window_error(fvecs, mask)
            calibscore = self.calibrator.score(modelscore.cpu())

        fused_score, p_cov, k_gain = self.kf.fuse(calibscore, ngscore, self.sigstg, self.sigdet)

        if draw:
            frame = self.detector.draw(frame, rvecs)
        self.results["scores"] = (modelscore, ngscore, calibscore)
        self.results["kf"] = (p_cov, k_gain)
        self.results["detng"] = (violations, ruleout)
        self.results["detect"] = (dets, vecs)
        return fused_score, frame

class RecursiveRiskDetector:  # TODO
    def __init__(self):
        pass