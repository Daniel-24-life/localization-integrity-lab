# Research Scope, Methodological Boundaries & Negative Results

To maintain rigorous scientific standards, this document explicitly records the empirical limitations, negative results, and boundaries of this research package.

---

## 1. Trade-Off Between Correlated Noise Modeling & White Noise Availability

The primary experimental insight from the 144-condition synthetic benchmark is a fundamental trade-off:
- **Under Correlated Noise (Matched Ornstein-Uhlenbeck Process)**:
  Augmenting the state vector with an estimated colored noise state eliminates Hazardously Misleading Information (reducing HMI duration from **88.3% down to 0.00%**).
- **Under Pure White Gaussian Noise**:
  The same colored-noise augmented filter experiences severe over-conservatism, causing Safe Availability to degrade from **99.6% down to 38.3%**, incurring a heavy false-alarm penalty.
- **Scientific Takeaway**:
  There is no "free lunch" in filter state augmentation. Robust integrity monitoring requires dynamic model switching or hypothesis banks rather than blindly expanding filter state dimensions.

---

## 2. Sample Size & Generalization Boundaries

1. **Synthetic Seeds vs. Independent Instances**:
   The 144 evaluations in the benchmark are generated from **8 unique random seeds across 2 trajectory types, 3 noise regimes, and 3 filter architectures** ($8 \times 2 \times 3 \times 3 = 144$). These 8 seeds serve as repeated trial units to assess variance; they do not constitute 144 independent real-world field trials.
2. **Real-World Generalization (G4 Gate)**:
   In preliminary real-robot indoor/outdoor evaluations under challenging wheel slip and GNSS multipath, true safe availability dropped to approximately **3.26%** under strict $0.5\,\text{m}$ Alert Limits. Industrial deployment certification and out-of-distribution sequence generalization remain unfulfilled research targets.
3. **Protection Level (PL) Risk Calibration**:
   The Protection Levels reported here are computed from linearized state covariance ellipsoids ($k_{\text{integrity}} \sqrt{\mathbf{P}_{11} + \mathbf{P}_{22}}$). They reflect theoretical confidence bounds under assumed Gaussianity and have not undergone empirical extreme-value tail fitting or formal aviation/automotive integrity risk allocation ($P_{\text{HMI}} < 10^{-7}$).

---

## 3. Data Provenance & Attribution

- Real-world sensor calibrations, lever-arm offsets, and camera-to-wheel transform conventions reference the publicly released M3DGR dataset guidelines (`calibration.md`).
- Synthetic trajectories are fully deterministic and self-contained within [`data/synthetic_generator.py`](file:///D:/AI文件/localization-integrity-lab/src/data/synthetic_generator.py), ensuring 100% reproducible execution on any standard workstation without downloading multi-gigabyte raw ROS bags.
