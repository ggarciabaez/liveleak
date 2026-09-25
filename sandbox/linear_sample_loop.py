import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from modules.stgnn import MockSTGNN
from sandbox.detng import *
from modules.kfilter import KalmanFilter

def application_loop(stgnn, detng, kf, T=5):
    feature_buffer = np.empty((0, 10, 13))

    # Store history as a dict for easier extraction
    hist = {
        't': deque(maxlen=50),
        'det': deque(maxlen=50),
        'gnn': deque(maxlen=50),
        'fused': deque(maxlen=50),
        'P': deque(maxlen=50),
        'K': deque(maxlen=50),
        'spike': deque(maxlen=50)
    }

    # ---------------------------------------------------------
    # UI SETUP: Subplots & Sliders
    # ---------------------------------------------------------
    plt.ion()  # Enable interactive mode
    fig, (ax_risk, ax_cov) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)  # Make room for sliders

    # Slider for ST-GNN Variance (Process Noise Q)
    ax_q = fig.add_axes([0.15, 0.1, 0.65, 0.03])
    slider_q = Slider(ax=ax_q, label='ST-GNN Var (Q)', valmin=0.001, valmax=1.0, valinit=0.05)

    # Slider for DetNG Variance (Measurement Noise R)
    ax_r = fig.add_axes([0.15, 0.05, 0.65, 0.03])
    slider_r = Slider(ax=ax_r, label='DetNG Var (R)', valmin=0.001, valmax=1.0, valinit=0.05)

    t = 0
    while True:
        t += 1

        # ---------------------------------------------------------
        # TIME: t+1 (New frame arrives)
        # ---------------------------------------------------------
        actual_fvecs = gen_random_data(1, 10, 1)[0]
        feature_buffer = np.concatenate([feature_buffer, actual_fvecs], axis=0)

        # 1. Evaluate independent models
        gnn_score = detng.execute(actual_fvecs)[0]

        stpred = stgnn(feature_buffer)
        pstd, vstd = np.std(stpred, axis=0).reshape(2, 2)
        # Add a tiny epsilon to prevent division by zero
        det_score = 1 - np.exp(-(np.sum(vstd) / (np.sum(pstd) + 1e-6)))

        # 2. Dynamic Noise Handling (from UI Sliders)
        stgnn_var = slider_q.val
        base_detng_var = slider_r.val

        # Simulate a tracking dropout for 3 frames every 50 frames
        is_spike = (t % 50 >= 47)
        # If camera is blocked/tracking drops, spike measurement noise 50x
        R_meas = base_detng_var * 50.0 if is_spike else base_detng_var

        # 3. FUSION
        fused_scores, P_cov, K_gain = kf.fuse(
            gnn_score,
            det_score,
            stgnn_variance=stgnn_var,
            detng_variance=R_meas
        )

        # Store data
        hist['t'].append(t)
        hist['det'].append(det_score)
        hist['gnn'].append(gnn_score)
        hist['fused'].append(fused_scores)
        hist['P'].append(float(P_cov))
        hist['K'].append(float(K_gain))
        hist['spike'].append(is_spike)

        if len(feature_buffer) == T:
            feature_buffer = feature_buffer[1:]

        # ---------------------------------------------------------
        # VISUALIZATION UPDATE
        # ---------------------------------------------------------
        if t % 2 == 0:
            # Clear axes but keep the figure and sliders intact
            ax_risk.clear()
            ax_cov.clear()

            # Formatting
            ax_risk.set_title("Risk Estimation Fusion", fontweight='bold')
            ax_risk.set_ylim(-0.05, 0.5)
            ax_risk.set_ylabel("Risk Probability")
            ax_risk.grid(True, linestyle='--', alpha=0.5)

            ax_cov.set_title("Filter State (Gain & Covariance)", fontweight='bold')
            ax_cov.set_ylim(-0.05, 0.5)
            ax_cov.grid(True, linestyle='--', alpha=0.5)

            # Extract lists
            tl = list(hist['t'])

            # Plot Risk
            ax_risk.plot(tl, list(hist['det']), 'r.', alpha=0.5, label="Raw DetNG")
            ax_risk.plot(tl, list(hist['gnn']), 'k--', alpha=0.6, label="ST-GNN Predict")
            ax_risk.plot(tl, list(hist['fused']), 'b-', linewidth=2.5, label="Fused Output")

            # Plot Covariance & Gain
            ax_cov.plot(tl, list(hist['P']), 'g-', linewidth=2, label="Uncertainty (P)")
            ax_cov.plot(tl, list(hist['K']), 'm-', linewidth=2, label="Kalman Gain (K)")

            # Shade regions where sensor dropout occurs
            for i, spike in enumerate(hist['spike']):
                if spike:
                    ax_risk.axvline(tl[i], color='red', alpha=0.1, linewidth=4)
                    ax_cov.axvline(tl[i], color='red', alpha=0.1, linewidth=4)

            ax_risk.legend(loc='upper right')
            ax_cov.legend(loc='upper right')

            plt.draw()
            plt.pause(0.01)


if __name__ == "__main__":
    stgnn = MockSTGNN()
    ng = DetNG().compile([
        (rule_ppe_compliance, 0.1),
        (rule_vectorized_proximity, 0.25),
        (rule_kinematic_ttc, 0.5),
        (rule_cone_zone, 0.15)
    ])
    kf = KalmanFilter(0.25, 0.25)
    application_loop(stgnn, ng, kf)