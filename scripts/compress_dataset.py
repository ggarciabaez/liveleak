# TODO: maybe outfit with argparse for paths?
from batchdet import BatchedDetector
from modules.dataset import Dataset
from glob import glob
import cv2
import numpy as np
import pickle

class VideoDataset:
    def __init__(self, path, glob_str="*.mp4", batch_size=8):
        self.path = path
        self.videos = glob(f"{self.path}/{glob_str}")
        self.vidid=0
        self.bs = min(batch_size, len(self.videos))

        self.curvid = None
        self._gen_vids()
        self.cover_cb = lambda: None

    def _gen_vids(self):
        def vidgen(src):
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file {src}")
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield frame
            finally:
                cap.release()
        self.curvid = [vidgen(self.videos[self.vidid+i]) for i in range(self.bs)]

    def __iter__(self):
        return self

    def __next__(self):  # TODO: batched videos? I'd have to get pretty knee deep into the detector, maybe bypass it.
        def cover(gen):
            try:
                return next(gen)
            except StopIteration:
                return self.cover_cb()

        frames = [cover(vid) for vid in self.curvid]
        if all(map(lambda x: x is None, frames)):
            # Yes. This is defined behavior.
            # We let the user handle shifting cause they might want to save the data onto their set
            # at that time. It's also a good idea to let them know when we're shifting, and letting
            # them control that is about the best way to do so.
            return None
        return frames

    def shift(self):
        self.vidid += self.bs
        if self.vidid >= len(self.videos):
            return False
        elif self.vidid + self.bs >= len(self.videos):
            self.bs = len(self.videos) - self.vidid
        self._gen_vids()
        return True

det = Detector("../assets/ppe_50ep.engine")
vdat = VideoDataset("../data/videos")  # TODO: find a dataset, damnit! Also it'd be good to have a dataset handler.
frameset: list[np.ndarray | None] | None = None
for frameset in vdat:  # this logic is pretty elegant but we still gotta try it.
    if frameset is None:
        # We gotta shift!
        if not vdat.shift():
            # We're done!
            break

        continue
    print([f.shape for f in frameset if f is not None])
