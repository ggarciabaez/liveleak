from modules.Detector import Detector
from modules.stgnn import STGNN
from modules.detng import DetNG, get_rules
from modules.kfilter import KalmanFilter
from collections import deque
from torch import load as tload
from tqdm import tqdm


class RiskDetector:
    def __init__(self, yolo_path, stgnn_path, winsize=5,
                 std_detng=0.1, std_stgnn=0.1,
                 graph_rad=250, ruleset=None):
        # The pipeline. quite simple, really.
        self.detector = Detector(yolo_path)
        self.detng = DetNG().compile(get_rules() if ruleset is None else ruleset)
        self.model = STGNN(graph_rad).load_state_dict(tload(stgnn_path))
        self.kf = KalmanFilter()
        self.results = {}

        self.qvec = deque(maxlen=winsize)
        self.sigdet = std_detng
        self.sigstg = std_stgnn

    def preload(self, frames):
        for frame in tqdm(frames):
            self.qvec.append(self.detector.get_vectors(frame))
            score, *_ = self.detng.execute(self.qvec)
            # Prime the KF
            self.kf.fuse(0.0, score, self.sigstg, self.sigdet)

    def __call__(self, frame, draw=False):
        dets = self.detector(frame)
        fvec = self.detector.get_vectors(dets)
        self.qvec.append(fvec)

        ngscore, violations, ruleout = self.detng.execute(self.qvec)
        modelscore = self.model(fvec)  # TODO: prep the model for passing lists! The embedding idea might still be best.
        fused_score, p_cov, k_gain = self.kf.fuse(modelscore, ngscore, self.sigstg, self.sigdet)
        if draw:
            frame = self.detector.draw(frame, dets, fvec)
        self.results["scores"] = (modelscore, ngscore)
        self.results["kf"] = (p_cov, k_gain)
        self.results["detng"] = (violations, ruleout)
        self.results["detect"] = (dets, fvec)
        return fused_score, frame

class RecursiveRiskDetector:  # TODO
    def __init__(self):
        pass