# Foundation Pit video sample

`foundation_pit_sample.zip` is a small, 24.3 MiB sample of three MP4 clips
from the Foundation Pit construction dataset. It is a preview/sample for the
repository, not the full training corpus.

The complete source archive is intentionally kept local at
`dataset_bundle/local_data/foundation_pit_construction.zip` and is ignored by
Git because it is about 4.2 GiB. Extract the desired video RAR files from that
archive into a local folder before preparing model features.

To prepare cached STGNN features after installing the project's Python
dependencies and obtaining the detector checkpoint, run from the repository
root:

```powershell
python -m modules.dataset --video-root <folder-with-extracted-videos> --detector-weights <path-to-yolo-checkpoint>
```

The default cache directory is `dataset_bundle/local_data/prepared/`, which
is also ignored by Git. One compressed `.npz` file is produced per source
video. The `Dataset` class in `modules/dataset.py` can then load train,
validation, or test windows. Its split is by video, and only persistent track
IDs present across an entire input/target window are retained. **No model was
trained as part of preparing this sample.**

When you're ready to train, use `scripts/train_stgnn.py` and choose a graph
radius in normalized image-coordinate units:

```powershell
python -m scripts.train_stgnn --features-dir dataset_bundle/local_data/prepared --graph-radius <radius>
```

The checkpoint is written under `dataset_bundle/local_data/` by default and
is not committed. Training needs a nonempty video-level validation split.

Source: [A Comprehensive Image and Video Dataset for Computer Vision-Based
Construction Safety Management](https://data.mendeley.com/datasets/xjm2czccx9/2),
Mendeley Data, version 2, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The three included clips are named `pexels-kelly-4376239-640x360-24fps.mp4`,
`pexels-free-videos-853867-1920x1080-25fps.mp4`, and
`pexels-arsel-ozgurdal-16628347-1920x1080-24fps.mp4` in the source archive.
