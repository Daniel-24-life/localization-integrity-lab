"""Explicit GPS week/TOW conversion. Leap offset must come from epoch evidence."""
import numpy as np

GPS_EPOCH_UNIX=315964800.


def gps_week_tow_to_utc_unix(week,tow,gps_minus_utc_sec):
    week,tow,offset=np.broadcast_arrays(np.asarray(week,dtype=float),np.asarray(tow,dtype=float),np.asarray(gps_minus_utc_sec,dtype=float))
    if not all(np.isfinite(x).all() for x in [week,tow,offset]) or np.any(week<0) or np.any(week!=np.floor(week)) or np.any(tow<0) or np.any(tow>=604800) or np.any(offset<0):
        raise ValueError('Invalid GPS week/TOW or explicit leap offset')
    return GPS_EPOCH_UNIX+week*604800+tow-offset
