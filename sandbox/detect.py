from ultralytics import YOLO
import supervision as sv
from trackers import OCSORTTracker as Tracker
import cv2

model = YOLO("yolo26s.pt", task="detect")
tracker = Tracker(30, 30)
cam = cv2.VideoCapture(0)

color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

box_annotator = sv.BoxAnnotator(
    color=color,
    color_lookup=sv.ColorLookup.TRACK)

label_annotator = sv.LabelAnnotator(
    color=color,
    color_lookup=sv.ColorLookup.TRACK,
    text_color=sv.Color.BLACK,
    text_scale=0.8)

trace_annotator = sv.TraceAnnotator(
    color=color,
    color_lookup=sv.ColorLookup.TRACK,
    thickness=2,
    trace_length=100)

while True:
    ret, frame = cam.read()
    result = model(frame)[0]

    det = sv.Detections.from_ultralytics(result)
    det = tracker.update(det)
    frame = box_annotator.annotate(scene=frame, detections=det)
    frame = trace_annotator.annotate(scene=frame, detections=det)
    frame = label_annotator.annotate(frame, det, det.tracker_id)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break