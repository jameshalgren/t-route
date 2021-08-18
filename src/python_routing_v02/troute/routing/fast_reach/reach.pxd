cdef struct QVD:
    double qdc
    double velc
    double depthc
    double cn
    double ck
    double X


cdef void muskingcunge(double dt,
        double qup,
        double quc,
        double qdp,
        double ql,
        double dx,
        double bw,
        double tw,
        double twcc,
        double n,
        double ncc,
        double cs,
        double s0,
        double velp,
        double depthp,
        QVD *rv) nogil

cpdef double[:,:] compute_reach(const double[:] boundary,
                                const double[:,:] previous_state,
                                const double[:,:] parameter_inputs,
                                double[:,:] output_buffer) nogil
