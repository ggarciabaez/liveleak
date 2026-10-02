import numpy as np

class DetNG:
    def __init__(self):
        self.rules = []
        self.weights = []
        # Feature indices based on [id, person, machine, vehicle, cone, px, py, vx, vy, hh, mask, vest, na]
        self.IDX = {
            'person': 1, 'machine': 2, 'vehicle': 3, 'cone': 4,
            'px': 5, 'py': 6, 'vx': 7, 'vy': 8,
            'hh': 9, 'mask': 10, 'vest': 11
        }
    @property
    def num_rules(self):
        return len(self.rules)

    def compile(self, rules_with_weights):
        """
        Compiles the engine with a list of tuples: (rule_function, weight)
        """
        self.rules, self.weights = zip(*rules_with_weights)
        self.rules = np.array(self.rules)
        # TODO: make a better weighting scheme s.t. weights are relative, not absolute.
        self.weights = np.array(self.weights)
        return self

    def execute(self, fvecs):
        """
        Executes rules over a (T, N, 13) feature buffer.
        Evaluates risk on the most recent frame T-1.
        """
        if len(self.rules) == 0:
            return 0.0, []

        # Extract latest frame (assume shape is T, N, 13)
        frames = np.atleast_3d(fvecs)
        full_ctx = []
        for i in range(len(fvecs)):
            current_frame = frames[i]

            # Filter valid entities (probability > 0.5)
            persons = current_frame[current_frame[:, self.IDX['person']] > 0.5]
            hazards = current_frame[(current_frame[:, self.IDX['machine']] > 0.5) |
                                    (current_frame[:, self.IDX['vehicle']] > 0.5)]
            cones = current_frame[current_frame[:, self.IDX['cone']] > 0.5]

            # Context dictionary passed to rules to prevent redundant filtering
            full_ctx.append({
                'frame': current_frame,
                'P': persons[:, [self.IDX['px'], self.IDX['py']]] if len(persons) else np.empty((0,2)),
                'V_P': persons[:, [self.IDX['vx'], self.IDX['vy']]] if len(persons) else np.empty((0,2)),
                'H': hazards[:, [self.IDX['px'], self.IDX['py']]] if len(hazards) else np.empty((0,2)),
                'V_H': hazards[:, [self.IDX['vx'], self.IDX['vy']]] if len(hazards) else np.empty((0,2)),
                'persons': persons,
                'cones': cones
            })

        # Execute all rules
        scores = np.array([rule(full_ctx) for rule in self.rules])
        
        # Weighted Soft Saturation Aggregation
        total_risk = 1 - np.exp(-np.sum(self.weights * scores))
        
        # Track which rules triggered violations (score > 0)
        violations = [r.__name__ for r in self.rules[scores > 0]]

        return total_risk, violations, scores


def gen_valid_feature(obj_type, state, ppe):
    """
    Create a valid feature vector.
    :param obj_type: A number [1, 4] determining the type of object.
    :param state: A 4-element array with the object's position and velocity [px, py, vx, vy].
    :param ppe: Integer bitmask for PPE (0-7).
    :return: 13-element numpy array.
    """
    v = np.zeros(13)
    v[obj_type] = 1.0
    v[5:9] = state

    if obj_type == 4:  # Cone
        v[7] = v[8] = 0.0  # Force zero velocity
        v[-1] = 1.0  # NA flag
    else:
        for i in range(3):
            v[9 + i] = ppe & 1
            ppe >>= 1
    return v


def gen_random_data(s=3, n=1, t=3):
    """
    Generate (S, T, N, 13) vectors of random, valid features.
    Simulates linear kinematics so temporal frames are continuous,
    then extracts them into sliding windows of length T.

    :param s: Number of sliding window samples
    :param n: Number of objects in each sample
    :param t: The time horizon (frames per sample)
    :return: Numpy array of shape (s, t, n, 13)
    """
    # Total unique frames needed to create 's' overlapping windows of length 't'
    total_frames = s + t - 1

    # Pre-allocate the continuous simulation buffer
    simulation = np.zeros((total_frames, n, 13))

    # 1. Initialize objects at Frame 0
    for i in range(n):
        obj_type = np.random.randint(1, 5)
        # State: [px, py, vx, vy] with positions (0-100) and velocities (-5 to 5)
        state = [
            np.random.uniform(0, 100), np.random.uniform(0, 100),
            np.random.uniform(-5, 5), np.random.uniform(-5, 5)
        ]
        ppe = np.random.randint(0, 8)  # 0 to 7 covers all 3 bits

        feat = gen_valid_feature(obj_type, state, ppe)
        feat[0] = i + 1  # Assign unique ID
        simulation[0, i] = feat

    # 2. Simulate forward kinematics for the remaining frames
    for f in range(1, total_frames):
        for i in range(n):
            prev_feat = simulation[f - 1, i].copy()
            # Update position: px = px + vx, py = py + vy
            prev_feat[5] += prev_feat[7]
            prev_feat[6] += prev_feat[8]
            simulation[f, i] = prev_feat

    # 3. Slice the continuous simulation into 's' overlapping samples of length 't'
    vecs = np.zeros((s, t, n, 13))
    for i in range(s):
        vecs[i] = simulation[i: i + t]

    return vecs

# ======= RULES ========
def rule_ppe_compliance(ctx):
    """Calculates risk based on missing Hard Hats or Vests."""
    ctx = ctx[-1]
    P = ctx['persons']
    if len(P) == 0:
        return 0.0
    # Risk increases if hh or vest is missing (values < 0.5)
    missing_hh = np.sum(P[:, 9] < 0.5)
    missing_vest = np.sum(P[:, 11] < 0.5)

    # Normalize by number of persons (max score 1.0)
    return min((missing_hh + missing_vest) / (len(P) * 2), 1.0)


def rule_vectorized_proximity(ctx, d_safe=10.0, d_danger=2.0):
    """Calculates D_{ij} and clips proximity risk."""
    ctx = ctx[-1]
    P, H = ctx['P'], ctx['H']
    if len(P) == 0 or len(H) == 0:
        return 0.0

    # Broadcasting to get pairwise differences (num_p, num_h, 2)
    diff = P[:, np.newaxis, :] - H[np.newaxis, :, :]
    D = np.linalg.norm(diff, axis=-1)

    # Calculate s_prox and return the maximum risk found in the frame
    s_prox = np.clip((d_safe - D) / (d_safe - d_danger), 0, 1)
    return np.max(s_prox)


def rule_kinematic_ttc(ctx, T_horizon=5.0, d_thresh=3.0, epsilon=1e-6):
    """Calculates Time-To-Collision and severity s_{ttc}."""
    ctx = ctx[-1]
    P, H = ctx['P'], ctx['H']
    V_P, V_H = ctx['V_P'], ctx['V_H']

    if len(P) == 0 or len(H) == 0:
        return 0.0

    # Relative positions (r) and velocities (v)
    r = P[:, np.newaxis, :] - H[np.newaxis, :, :]
    v = V_P[:, np.newaxis, :] - V_H[np.newaxis, :, :]

    # Dot products
    r_dot_v = np.sum(r * v, axis=-1)
    v_sq = np.sum(v * v, axis=-1) + epsilon

    # Time to closest approach
    t_star = -r_dot_v / v_sq

    # Converging pairs condition (r_dot_v < 0) & valid time horizon
    valid_mask = (r_dot_v < 0) & (t_star > 0) & (t_star <= T_horizon)

    if not np.any(valid_mask):
        return 0.0

    # Predicted closest distance for all valid pairs
    d_min = np.linalg.norm(r + v * t_star[..., np.newaxis], axis=-1)

    # Calculate severity only where d_min < d_thresh and valid_mask is True
    danger_mask = valid_mask & (d_min < d_thresh)

    if not np.any(danger_mask):
        return 0.0

    s_ttc = (1 - (t_star[danger_mask] / T_horizon)) * (1 - (d_min[danger_mask] / d_thresh))
    return np.max(s_ttc)


def rule_cone_zone(ctx, buffer=1.0):
    """Basic bounding box check for person within cone coordinates."""
    ctx = ctx[-1]
    P = ctx['P']
    C = ctx['cones'][:, [5, 6]] if len(ctx['cones']) else np.empty((0, 2))

    if len(P) == 0 or len(C) < 3:
        return 0.0  # Need at least 3 cones to form a meaningful zone

    x_min, y_min = np.min(C, axis=0) - buffer
    x_max, y_max = np.max(C, axis=0) + buffer

    # Check if any person is inside the bounding box of the cones
    in_zone = (P[:, 0] > x_min) & (P[:, 0] < x_max) & (P[:, 1] > y_min) & (P[:, 1] < y_max)

    return 1.0 if np.any(in_zone) else 0.0


def get_rules(weights=(0.1, 0.25, 0.5, 0.15)):
    return tuple(zip([rule_ppe_compliance, rule_vectorized_proximity, rule_kinematic_ttc, rule_cone_zone],
                     weights))