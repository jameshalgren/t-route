cdef double simple_da_with_decay(
    const double last_valid_obs,
    const double model_val,
    const double minutes_since_last_valid,
    const double decay_coeff,
) nogil


cdef double obs_persist_shift(
    const double last_valid_obs,
    const double model_val,
    const double minutes_since_last_valid,
    const double decay_coeff,
) nogil
