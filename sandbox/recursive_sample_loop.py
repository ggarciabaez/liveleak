from collections import deque
from modules.stgnn import MockSTGNN
from sandbox.chained_fusion import RuleScoreKalmanFilter
from sandbox.detng import *
from matplotlib import pyplot as plt
from collections import deque

def application_loop(stgnn: MockSTGNN, detng: DetNG, kf: RuleScoreKalmanFilter, T=5):
    # History buffer of length T for the ST-GNN
    feature_buffer = np.empty((0, 10, 13))
    risk_hist = deque(maxlen=50)
    cov_hist = deque(maxlen=50)
    # Stores the prediction made at time t, to be used at time t+1
    predicted_rule_scores = np.zeros(detng.num_rules)
    t = 0
    fig, (riskp, pp) = plt.subplots(1, 2)
    while True:
        riskp.clear()
        pp.clear()
        print(f"\nTime: {t}")
        t += 1
        # ---------------------------------------------------------
        # TIME: t+1 (New frame arrives)
        # ---------------------------------------------------------
        actual_fvecs = gen_random_data(1, 10, 1)[0]  # (1, 10, 13)
        feature_buffer = np.concatenate([feature_buffer, actual_fvecs], axis=0)

        # 1. MEASUREMENT (DetNG processes the raw new frame)
        _, _, measured_scores = detng.execute(actual_fvecs)

        # Determine current measurement noise R (e.g., higher if tracking confidence is low)
        base_R = np.array([0.05, 0.05, 0.05, 0.05])

        # Simulate a "sensor failure" or "flickering bounding box" on frame 25
        if t%25==0:
            base_R = np.random.normal(base_R, 5.0)  # Huge uncertainty spike on TTC measurement

        R_meas = np.diag(base_R)

        # 2. FUSION (Kalman Filter blends last step's prediction with today's measurement)
        fused_scores, P_cov, K_gain = kf.fuse(x_pred=predicted_rule_scores, R_meas=R_meas, measured_scores=measured_scores)

        # 3. FINAL OUTPUT PROBABILITY (Soft Saturation Aggregation)
        # We apply the weights and the exponential function to the FUSED scores
        total_risk = 1 - np.exp(-np.sum(detng.weights * fused_scores))
        d = np.diag(P_cov.copy())
        print(f"Risk & P: {total_risk:.3f} {d}")
        risk_hist.append((t, total_risk))
        cov_hist.append(d)

        # ---------------------------------------------------------
        # TIME: PREPARING FOR t+2
        # ---------------------------------------------------------
        if len(feature_buffer) == T:
            # 4. PREDICT NEXT STATE KINEMATICS (ST-GNN)
            # ST-GNN looks at the historical buffer [t-T+1 ... t+1]
            predicted_kinematics = stgnn(feature_buffer)  # Outputs [px, py, vx, vy] for t+2

            # 5. PREDICT NEXT STATE RISK (DetNG)
            # Create a synthetic feature vector for t+2
            # We copy the current frame and overwrite the kinematics with the ST-GNN predictions
            synthetic_fvecs_t2 = np.roll(feature_buffer, -1, axis=0)
            synthetic_fvecs_t2[-1, :, 5:9] = predicted_kinematics

            # DetNG evaluates the ST-GNN's predicted future
            _, _, predicted_rule_scores = detng.execute(synthetic_fvecs_t2)
            feature_buffer = feature_buffer[1:]

        if t % 2 == 0:  # Only redraw every other frame to save CPU
            riskp.clear()
            pp.clear()
            riskp.set_title("Total Risk")
            pp.set_title("Filter Uncertainty (P)")

            lt, rt = zip(*risk_hist)
            ct = np.array(cov_hist.copy()).T

            riskp.plot(lt, rt, color='red')
            riskp.set_ylim(-0.1, 1.1)

            for i in range(len(ct)):
                pp.plot(lt, ct[i], label=f"Rule {i} Uncert.")

            pp.legend(loc='upper right')
            plt.pause(0.01)


if __name__ == "__main__":
    stgnn = MockSTGNN()
    ng = DetNG().compile([
        (rule_ppe_compliance, 0.1),
        (rule_vectorized_proximity, 0.25),
        (rule_kinematic_ttc, 0.5),
        (rule_cone_zone, 0.15)
    ])
    kf = RuleScoreKalmanFilter(ng.num_rules, procnoise=[0.01]*ng.num_rules)
    application_loop(stgnn, ng, kf)