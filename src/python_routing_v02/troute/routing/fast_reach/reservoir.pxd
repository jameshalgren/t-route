cdef struct QH:
    double resoutflow
    double reslevel

cdef void levelpool_physics(double dt,
        double qi0,
        double qi1,
        double ql,
        double ar,
        double we,
        double maxh,
        double wc,
        double wl,
        double dl,
        double oe,
        double oc,
        double oa,
        double H0,
        QH *rv) nogil

cpdef double[:,:] compute_reservoir(const double[:] boundary,
                                    const double[:,:] previous_state,
                                    const double[:,:] parameter_inputs,
                                    double[:,:] output_buffer) nogil
