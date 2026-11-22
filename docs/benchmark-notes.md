# Benchmark notes

The public pipeline compares model and filter behavior under a fixed synthetic design. Start with the [configuration](../configs/default_config.yaml), run the [synthetic study](../scripts/run_synthetic_study.py), and read the aggregate outputs alongside the [scope and limitations](research_limitations.md). Each seed is reused across multiple settings, so aggregation should preserve the pairing.

The research also compares a colored-noise model with a persistence baseline. Small differences and weakly identified parameters matter for model interpretation. A result from the absorbing-boundary model is a first-exit result for the specified stochastic process. The independent field-data evaluation requires valid observations and reference alignment. The available independent sequence did not qualify for scoring in the subsequent research record. Protection-level risk calibration and the real-data G4 gate remain open.
