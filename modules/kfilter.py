"""
The Kalman filter to act as bottleneck
"""

class KalmanFilter:
    def __init__(self, process_noise=0.01, initial_err_estimate=0.01):
        self.Q = process_noise
        self.P = initial_err_estimate  # covariance
        self.x = 0.0

    def fuse(self, stgnn_risk, detng_risk, stgnn_variance, detng_variance):
        """
        Fuses the two risk estimates into a single one.
        The stgnn risk is used as prediction, and the detngn risk is used as update.
        :param stgnn_risk:
        :param detng_risk:
        :param stgnn_variance:
        :param detng_variance:
        :return:
        """

        P_pred = stgnn_variance + self.Q  # variance in prediction + process noise
        K = P_pred / (P_pred + detng_variance)  # Kalman gain
        self.x = stgnn_risk + K * (detng_risk-stgnn_risk)
        self.P = (1-K) * P_pred
        return self.x, self.P, K


if __name__ == '__main__':
    import numpy as np
    kf = KalmanFilter()
    for i in range(10):
        x, P, K = kf.fuse(np.random.rand(), np.random.rand(), 0.2, 0.5)
        print(f"Predicted risk: {x:.3f} | Covariance: {P:.3f} | Kalman Gain: {K:.3f}")