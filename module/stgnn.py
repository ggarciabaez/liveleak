"""Planned ST-GNN forecasting and anomaly-scoring module.

This file is a commented outline only. It does not train a model or produce
predictions yet. The shared node and edge schema must be agreed on before the
interfaces below are implemented.
"""


"""Receive entity vectors
Take the person, machinery, vehicle, and cone vectors produced by the
detector. Each vector already contains an ID, type, position, velocity,
and PPE information
"""


"""Build the graph for each frame
Combine the entity vectors into a node-feature matrix, keeping IDs
separately. Use node positions to find nearby pairs. For each connection,
record the relative position (dx, dy) and distance between the two nodes.
Share this graph with the rule and ST-GNN branches.
"""

"""Collect a short history
Use track IDs to connect the same entities across several frame graphs.
Keep a sliding window and mark entities that enter, leave, or disappear.
"""

"""Forecast future motion
Use the graph relationships and each entity's history to predict its
position and velocity over the next few frames.
"""

"""Measure prediction error
When the predicted frames arrive, compare each valid forecast with the
observed position and velocity. A larger error indicates less typical
motion; it does not by itself prove a safety risk.
"""

"""Calibrate and pass on the score
Use held-out normal footage to map prediction errors to a 0-1 anomaly
score. Send an available score to Kalman fusion alongside the separate
rule score. Report no ST-GNN score until a forecast can be checked.
"""