from modules.detng import *

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

def test_performance(ng):
    from time import perf_counter
    data = gen_random_data(s=100, n=10, t=5)
    start = perf_counter()
    scorelists = np.empty((len(data), len(ng.rules)))
    for i, s in enumerate(data):
        scorelists[i] = ng.execute(s)[2]
    end = perf_counter()
    processed_score = 1 - np.exp(-np.sum(ng.weights[np.newaxis, ...] * scorelists, axis=0))

    print(f"Execution time: {1 / (end - start):.3f} seconds")
    print(f"Scores: {np.unique(scorelists, axis=0)}")
    print(f"Processed Score: {processed_score}")
    print([r.__name__ for r in ng.rules])

if __name__ == "__main__":
    # Compile with rules and their respective weights w_k
    ng = DetNG().compile([
        (rule_ppe_compliance, 0.1),
        (rule_vectorized_proximity, 0.25),
        (rule_kinematic_ttc, 0.5),
        (rule_cone_zone, 0.15)
    ])

    # Mock buffer: 1 time step, 3 objects, 13 features
    # Format: [id, p, m, v, c, px, py, vx, vy, hh, mask, vest, na]
    mock_fvecs = np.array([[
        [1, 1, 0, 0, 0, 10, 10, 2, 0, 1, 1, 1, 0], # Person moving Right
        [2, 0, 1, 0, 0, 15, 10, -3, 0, 0, 0, 0, 0], # Machine moving Left (Converging)
        [3, 0, 0, 0, 1, 20, 20, 0, 0, 0, 0, 0, 0],  # Irrelevant Cone
    ]])
    for i in range(3):
        mock_fvecs = np.concatenate([mock_fvecs, mock_fvecs], axis=0)
    print(mock_fvecs.shape)
    score, violations, scorelist = ng.execute(mock_fvecs)
    print(f"Risk Score: {score:.3f}")
    print(f"Violations: {violations}")
    print(f"Scores: {scorelist}\n")

    # test_performance(ng)
