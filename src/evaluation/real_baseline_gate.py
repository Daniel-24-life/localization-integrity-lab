"""Recompute numerical baseline eligibility; never certify reference accuracy."""
import numpy as np


def assess_real_baseline(quality, protocol, metadata, rows, jacobian_error):
    values = quality['whole_horizon']['values']
    rules = protocol['quality_rules']
    # Do not accept a manually edited quality.passed boolean without the values.
    numeric_quality = all(k in rules and np.isfinite(v) and v <= rules[k] for k, v in values.items())
    required = {k for k, v in rules.items() if isinstance(v, (float, int)) and not isinstance(v, bool)}
    numeric_quality = numeric_quality and required.issubset(values)
    finite_source = quality['whole_horizon']['checks'].get('finite_strictly_increasing_source_data') is True
    provenance_eligible = (metadata.get('reference_status') == 'available'
        and metadata.get('reference_independent') is True
        and metadata.get('evidence_class') == 'real_reference_candidate'
        and bool(metadata.get('reference_source')) and bool(metadata.get('position_observation_source')))
    whole = {r['model']: r for r in rows if r['window'] == 'whole'}
    dr, fixed = float(whole['B0']['ate_rmse']), float(whole['B1']['ate_rmse'])
    improvement = 100. * (1. - fixed / dr) if np.isfinite([dr, fixed]).all() and dr > 0 and fixed >= 0 else None
    eligible = bool(numeric_quality and finite_source and provenance_eligible)
    numerical = improvement is not None and improvement > 50. and np.isfinite(jacobian_error) and 0 <= jacobian_error < 1e-6
    return dict(real_scope_eligible=eligible, real_G2_numeric_pass=bool(eligible and numerical),
        improvement_percent=improvement, independent_accuracy_or_integrity_certification=False,
        scope='Predeclared whole-horizon benchmark baseline only; reference provenance still requires source review')
