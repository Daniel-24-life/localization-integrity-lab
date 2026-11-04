"""Traceable canonical ordering, never smoothing/replacing original reference rows."""
import numpy as np


def canonicalize_reference_rows(rows,max_backward_sec=5e-5):
    rows=np.asarray(rows,dtype=float)
    if rows.ndim!=2 or rows.shape[1]!=8 or len(rows)<2 or not np.isfinite(rows).all():
        raise ValueError('Expected finite nonempty TUM-style eight-column reference')
    dt=np.diff(rows[:,0])
    if not np.isfinite(max_backward_sec) or max_backward_sec<0 or np.any(dt < -max_backward_sec):
        raise ValueError('Reference reversal exceeds declared timestamp-jitter allowance')
    order=np.argsort(rows[:,0],kind='stable');sorted_rows=rows[order]
    unique,first,count=np.unique(sorted_rows[:,0],return_index=True,return_counts=True)
    for i,n in zip(first,count):
        if n>1 and not np.all(sorted_rows[i:i+n,1:]==sorted_rows[i,1:]):
            raise ValueError('Conflicting poses at an identical timestamp; manual review required')
    if len(unique)<2:raise ValueError('Insufficient distinct reference times')
    retained_indices=order[first]
    return rows[retained_indices].copy(),dict(original_rows=len(rows),canonical_rows=len(unique),
        exact_duplicate_rows_removed=len(rows)-len(unique),original_negative_intervals=int(np.sum(dt<0)),
        largest_original_reversal_sec=float(max(0.,-dt.min())),max_backward_sec=max_backward_sec,
        retained_original_row_indices=retained_indices,position_or_quaternion_values_changed=False,
        policy='stable timestamp order within declared jitter; collapse only exactly identical timestamp+pose rows')
