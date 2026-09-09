from ultralytics import YOLO
import cv2

cam = cv2.VideoCapture(0)
model = YOLO("./assets/abombinmycar.engine")

while True:
    ret, frame = cam.read()
    results = model(frame)
    for result in results:
        frame = result.plot(img=frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



"""
#Import everything important
from modules.Detector import Detector
from modules.tracker import Tracker
from modules.info_vector import InfoVectorBuilder
from modules.graph import GraphBuilder
from modules.rules import RuleEngine
from modules.stgnn import STGNNForecaster
from modules.kfilter import KalmanFilter
from modules.visualizer import Visualizer
"""

def main():
    """
    # Initialize components
    detector = Detector()
    tracker = Tracker()
    info_builder = InfoVectorBuilder()
    graph_builder = GraphBuilder()
    rule_engine = RuleEngine()
    # stgnn = ST-GNN model initialization
    kalman = KalmanFilter()
    visualizer = Visualizer()

    
    # Video Input
    cam = cv2.VideoCapture(0)

    
    # Process every frame
    for frame in video:

        # YOLO26m
        detections = detector.detect(frame)

        # Tracking
        tracks = tracker.update(detections)

        # Convert tracks into standardized nodes
        nodes = info_builder.build(tracks)

        # Build spatial graph 
        graph_window = graph_builder.update(nodes)

        
        #BRANCHES
        # Rule-based branch
        rule_score = rule_engine.evaluate(graph_window)
        # ST-GNN branch
        gnn_probability = stgnn.predict(graph_window)


        # Fuse both
        risk_score = kalman.update(
            rule_score,
            gnn_probability
        )

        # Visualization
        visualizer.draw(something)
"""

if __name__ == "__main__":
    main()