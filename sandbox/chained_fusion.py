import numpy as np


class RuleScoreKalmanFilter:
    def __init__(self, num_rules, procnoise=None, init_std=1.0):
        self.num_rules = num_rules
        self.x = np.zeros(num_rules)
        self.P = np.eye(num_rules) * init_std  # Estimate Covariance (Starts highly uncertain)

        # Q: Process Noise.
        # e.g., PPE risk changes slowly (low Q), TTC changes instantly (high Q)
        if procnoise is not None:
            assert len(procnoise) == num_rules, ()
            self.Q = np.diag(procnoise)
        else:
            self.Q = np.eye(num_rules) * 0.05

    def fuse(self, x_pred, measured_scores, R_meas):
        """
        Fuses ST-GNN predicted risk with DetNG measured risk.

        :param x_pred: (num_rules,) array of risk scores from DetNG(ST-GNN_kinematics)
        :param measured_scores: (num_rules,) array of risk scores from DetNG(raw_frame)2
        :param R_meas: (num_rules, num_rules) Covariance matrix of the current measurement.
                       Can be dynamically scaled if current frame has tracking dropouts.
        """
        self.P += self.Q  # Add noise every timestep

        # H is Identity, so K = P * (P + R)^-1
        K_gain = self.P @ np.linalg.inv(self.P + R_meas)
        self.x = x_pred + K_gain @ (measured_scores - x_pred)
        self.P = (np.eye(self.num_rules) - K_gain) @ self.P  # Update error covariance

        return self.x, self.P, K_gain


if __name__ == "__main__":
    # Example scenario with 4 rules
    kf = RuleScoreKalmanFilter(num_rules=4)

    # DetNG scores based on ST-GNN predicted kinematics
    predicted_scores = np.array([0.1, 0.4, 0.8, 0.0])

    # DetNG scores based on current raw frame
    measured_scores = np.array([0.1, 0.3, 0.2, 0.0])

    for i in range(5):
        R = np.eye(4) * np.random.rand()/10  # Object detector trust
        x, cov, K = kf.fuse(predicted_scores, measured_scores, R)
    print("Final Fused Risk Scores:", np.round(x, 3))
    print(f"Covariance Matrix:\n{cov}")
    print(f"Kalman Gain Matrix:\n{K}")
    print("Influence (Diagonal of Kalman Gain K):")
    for i, rule in enumerate(['PPE', 'Prox', 'TTC', 'Cone']):
        # K -> 1.0 means DetNG measurement is trusted
        # K -> 0.0 means ST-GNN prediction is trusted
        print(f"  {rule}: {K[i, i]:.3f}")