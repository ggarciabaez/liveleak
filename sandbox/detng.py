from modules.detng import *



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
