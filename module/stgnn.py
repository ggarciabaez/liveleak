"""ST-GNN pipeline outline.

This file describes the intended flow only. It contains no model code yet.
"""


"""1. Receive entity vectors.

Take the person, machinery, vehicle, and cone vectors produced by the
detector. Each vector already contains an ID, type, position, velocity,
and PPE information.
"""


"""2. Build a graph for each frame.

Treat detected entities as nodes. Connect nearby entities so the model
can learn how their movements relate to one another.
"""


"""3. Collect a short history.

Use track IDs to follow the same entities across several frames. Keep
their graph information in a sliding time window.
"""


"""4. Forecast future motion.

Use a spatial graph model and a temporal model to predict where each
tracked entity will be and how it will move in upcoming frames.
"""


"""5. Measure prediction error.

When those future frames arrive, compare the forecast with the observed
positions and velocities. Larger errors suggest less typical motion.
"""


"""6. Calibrate and pass on the score.

Map prediction error to a 0-1 anomaly score using normal validation
footage, then pass that score to the Kalman fusion stage. If there is not
enough history or no future observation yet, report that no score is ready.
"""
