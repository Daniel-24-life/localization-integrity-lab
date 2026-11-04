"""Plots label evidence class, reference, thresholds and alarm semantics."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_evaluation(path, data, dr_state, result, integrity, alert_limit, nis_threshold=9.21):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    a, b, c, d = axes.flat
    a.plot(data.gt_x, data.gt_y, "k-", label="Evaluation reference")
    a.plot(dr_state[:, 0], dr_state[:, 1], "g--", alpha=.6, label="Wheel DR")
    a.plot(result["state"][:, 0], result["state"][:, 1], "b-", label="Fixed EKF")
    a.set(xlabel="X (m)", ylabel="Y (m)", title="Trajectory (reference-initialized)")
    a.axis("equal")
    t = data.timestamp
    b.plot(t, integrity["true_errors"], "r-", label="Error to reference")
    b.plot(t, result["pl"], "b--", label="Covariance PL estimate")
    b.axhline(alert_limit, color="k", linestyle=":", label=f"AL={alert_limit:g} m")
    b.fill_between(t, 0, 1, where=integrity["hmi_mask"], transform=b.get_xaxis_transform(), alpha=.15, color="orange", label="Instantaneous HMI samples")
    b.set(xlabel="Time (s)", ylabel="m", title="Instantaneous alarm, no hold")
    ref_speed = np.linalg.norm(np.diff(np.column_stack([data.gt_x,data.gt_y]),axis=0),axis=1)/np.diff(t)
    c.plot(t[:-1],ref_speed,"k-",label="Reference speed magnitude (unsmoothed)")
    c.plot(t,data.wheel_v,"r--",label="Signed wheel speed")
    c.set(xlabel="Time (s)",ylabel="m/s",title="Magnitude and signed speed differ in reverse")
    d.plot(t,result["nis"],"m.",markersize=2,label="NIS (missing stays NaN)")
    d.axhline(nis_threshold,color="r",linestyle="--",label=f"Threshold={nis_threshold:g}")
    d.set(xlabel="Time (s)",ylabel="NIS",title="NIS available only at updates")
    for ax in axes.flat:
        ax.legend(fontsize=7)
        ax.grid(alpha=.25)
    fig.suptitle(f'{data.metadata.get("evidence_class", "unknown")} | exploratory diagnostics; alignment not certified', fontsize=11)
    fig.savefig(path,dpi=150)
    plt.close(fig)
