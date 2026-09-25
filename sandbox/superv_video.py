from ultralytics import YOLO
import supervision as sv
from supervision.assets import download_assets, VideoAssets
from trackers import BoTSORTTracker
from tqdm import tqdm

# Download a supervision video asset
path_to_video = download_assets(VideoAssets.PEOPLE_WALKING)
video_info = sv.VideoInfo.from_video_path(path_to_video)

label = sv.LabelAnnotator()
byte_tracker = BoTSORTTracker()
model = YOLO("../assets/ppe_50ep.engine", task="detect")

# Create a frame generator from video path for iteration of frames.
frame_generator = sv.get_video_frames_generator(path_to_video)

# Create a video sink context manager to save resulting video.
with sv.VideoSink(target_path="output.mp4", video_info=video_info) as sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        result = model.predict(frame, quantize=16)[0]

        # Convert model results to a supervision detection object.
        detections = sv.Detections.from_ultralytics(result)

        tracked_detections = byte_tracker.update(detections)
        print(tracked_detections)
        # Create labels with tracker_id for label annotator.
        labels = [ f"{tracker_id}" for tracker_id in tracked_detections.tracker_id ]

        # Apply label annotator to frame.
        annotated_frame = label.annotate(scene=frame.copy(), detections=tracked_detections, labels=labels)
        sink.write_frame(annotated_frame)