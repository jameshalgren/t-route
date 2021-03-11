# cython: language_level=3, cdivision=True

cdef extern from "<math.h>" nogil:
    double fabs(double x)
    float fabsf(float)
    double fmin(double x, double y)
    float fminf(float, float)
    double fmax(double x, double y)
    float fmaxf(float, float)
    double sqrt(double x)
    float sqrtf(float)
    double pow(double x, double y)
    float powf(float, float)


cdef struct QHC:
    float h
    float Q_mc
    float Q_normal
    float Q_j
    float Xmin
    float X
    float Km
    float D
    float ck
    float cn
    float C1
    float C2
    float C3
    float C4


cdef struct channel_properties:
    float bfd
    float bw
    float tw
    float twcc
    float z
    float s0
    float sqrt_s0
    float sqrt_1z2
    float n
    float ncc


cdef struct hydraulic_geometry:
    float twl
    float R
    float AREA
    float AREAC
    float WP
    float WPC
    float h_lt_bf
    float h_gt_bf


# INPUTS:
# Arrays:
# Q: np.array([qup, quc, qdp, 1])
# C: np.array([C1, C2, C3, C4])
# dx, bw, tw, twcc, ncc, cs, so, n, qlat: 1D arrays
# vela, deptha: 2D arrays
# qd: 3D arrays

# Scalars:
# dt, qup, quc, qdp, qdc, ql, z, vel, depth: float
# bfd, WPC, AREAC: float
# ntim: integer
# ncomp, linkID: integer

# dt = 60.0 # Time step
# dx = 1800.0 # segment length
# bw = 112.0 # Trapezoidal bottom width
# tw = 448.0 # Channel top width (at bankfull)
# twcc = 623.5999755859375 # Flood plain width
# n = 0.02800000086426735 # manning roughness of channel
# ncc = 0.03136000037193298 # manning roughness of floodplain
# cs = 1.399999976158142 # channel trapezoidal sideslope
# s0 = 0.0017999999690800905 # downstream segment bed slope
# ql = 40.0 # Lateral inflow in this time step
# qup = 0.04598825052380562 # Flow from the upstream neighbor in the previous timestep
# quc = 0.04598825052380562 # Flow from the upstream neighbor in the current timestep
# qdp = 0.21487340331077576 # Flow at the current segment in the previous timestep
# depthp = 0.010033470578491688 # Depth at the current segment in the previous timestep')
# Expected (0.7570107129107292, 0.12373605608067537, 0.023344515869393945)

#args = (60.0, 0.04598825052380562, 0.04598825052380562, 0.21487340331077576,
 #       40.0, 1800.0, 112.0, 448.0, 623.5999755859375, 0.02800000086426735,
 #       0.03136000037193298, 1.399999976158142, 0.0017999999690800905, 0.010033470578491688)
#
#single_precision = tuple(map(np.float32, args))
#double_precision = tuple(map(np.float64, args))

# single_seg.muskingcungenwm(dt, qup, quc, qdp, ql, dx, bw, tw, twcc, n, ncc, cs, s0, depthp)


cdef void cython_muskingcunge(
    const float dt,
    const float qup,
    const float quc,
    const float qdp,
    const float ql,
    const float dx,
    const float bw,
    const float tw,
    const float twcc,
    const float n,
    const float ncc,
    const float cs,
    const float s0,
    const float velp,
    const float depthp,
    QVD *rv,
) nogil:
    cdef int maxiter = 100
    cdef float mindepth = 0.01
    cdef float aerror = 0.01
    cdef float rerror = 1.0
    cdef int it, tries = 0

    cdef float h, h_0
    cdef float h_1
    cdef float qdc, velc, depthc
    cdef float C_pdot_Q
    cdef float R, twl
    cdef float bfd
    cdef QHC qc_struct_left
    cdef QHC qc_struct_right
    cdef channel_properties chan_struct

    # populate channel properties
    chan_struct.n = n
    chan_struct.ncc = ncc
    chan_struct.bw = bw
    chan_struct.tw = tw
    chan_struct.twcc = twcc

    chan_struct.s0 = s0
    chan_struct.sqrt_s0 = sqrtf(chan_struct.s0)

    if cs == 0:
        chan_struct.z = 1
    else:
        chan_struct.z = 1/cs

    chan_struct.sqrt_1z2 = sqrtf(1.0 + (chan_struct.z * chan_struct.z))

    if chan_struct.bw > chan_struct.tw:
        chan_struct.bfd = chan_struct.bw * (1/0.00001)
    elif chan_struct.bw == chan_struct.tw:
        chan_struct.bfd = chan_struct.bw/(2*chan_struct.z)
    else:
        chan_struct.bfd = (chan_struct.tw - chan_struct.bw)/(2*chan_struct.z)

    cdef channel_properties *chan = &chan_struct

    # initialize vars
    qc_struct_left.h = depthp * 0.67
    qc_struct_left.Q_mc = 0.0
    qc_struct_left.Q_normal = 0.0
    qc_struct_left.Q_j = 0.0
    qc_struct_left.Xmin = 0.0
    qc_struct_left.X = 0.0
    qc_struct_left.Km = 0.0
    qc_struct_left.D = 0.0
    qc_struct_left.ck = 0.0
    qc_struct_left.cn = 0.0
    qc_struct_left.C1 = 0.0
    qc_struct_left.C2 = 0.0
    qc_struct_left.C3 = 0.0
    qc_struct_left.C4 = 0.0

    qc_struct_right.h = (depthp * 1.33) + mindepth
    qc_struct_right.Q_mc = 0.0
    qc_struct_right.Q_normal = 0.0
    qc_struct_right.Q_j = 0.0
    qc_struct_right.Xmin = 0.25
    qc_struct_right.X = 0.0
    qc_struct_right.Km = 0.0
    qc_struct_right.D = 0.0
    qc_struct_right.ck = 0.0
    qc_struct_right.cn = 0.0
    qc_struct_right.C1 = 0.0
    qc_struct_right.C2 = 0.0
    qc_struct_right.C3 = 0.0
    qc_struct_right.C4 = 0.0

    cdef QHC *qc_left = &qc_struct_left
    cdef QHC *qc_right = &qc_struct_right

    cdef hydraulic_geometry hg_struct
    cdef hydraulic_geometry *hg = &hg_struct

    cdef float C1, C2, C3, C4
    cdef float Qj_0, Qj

    depthc = fmaxf(depthp, 0)

    if ql > 0 or quc > 0 or qup > 0 or qdp > 0:

        it = 0

        while rerror > 0.01 and aerror >= mindepth and it <= maxiter:
            compute_mc_flow(
                chan,
                dt,
                dx, qup, quc, qdp, ql,
                hg,
                qc_left,
            )

            qc_right.Q_j = qc_left.Q_mc

            compute_mc_flow(
                chan,
                dt,
                dx, qup, quc, qdp, ql,
                hg,
                qc_right,
            )

            Qj_0 = qc_left.Q_j
            Qj = qc_right.Q_j
            h_0 = qc_left.h
            h = qc_right.h

            if Qj_0 - Qj != 0:
                h_1 = h - (Qj * (h_0 - h)) / (Qj_0 - Qj)

                if h_1 < 0:
                    h_1 = h
            else:
                h_1 = h

            if h > 0:
                rerror = fabsf((h_1 - h) * (1/h))
                aerror = fabsf(h_1 - h)
            else:
                rerror = 0
                aerror = 0.9

            h_0 = max(0, h)
            h = max(0, h_1)
            it += 1
            qc_left.h = h_0
            qc_right.h = h

            if h < mindepth:
                if it >= maxiter:
                    tries += 1
                    if tries <= 4:
                        h = h * 1.33
                        h_0 = h_0 * 0.67
                        maxiter += 25
                        Qj_0 = 0
                        WPC = 0
                        AREAC = 0
                        _iter = 0
                        continue
                    else:
                        break

        C1 = qc_struct_right.C1
        C2 = qc_struct_right.C2
        C3 = qc_struct_right.C3
        C4 = qc_struct_right.C4

        C_pdot_Q = (C1 * qup) + (C2 * quc) + (C3 * qdp)
        qdc = C_pdot_Q + C4
        if qdc < 0:
            if C4 < 0 and fabsf(C4) > C_pdot_Q:
                qdc = 0
            else:
                qdc = max((C1 * qup) + (C2 * quc) + C4, (C1 * qup) + (C3 * qdp) + C4)

        twl = chan_struct.bw + (2 * chan_struct.z * h)
        R = (0.5 * h * (chan_struct.bw + twl)) / (chan_struct.bw + 2.0 * sqrtf((((twl - chan_struct.bw) * 0.5) ** 2.0) + (h * h)))
        velc = (1 / chan_struct.n) * ((R ** 2.0/3.0)) * chan_struct.sqrt_s0
        depthc = h
    else:
        qdc = 0
        depthc = 0
        velc = 0

    rv.qdc = qdc
    rv.velc = velc
    rv.depthc = depthc


cdef inline void compute_mc_flow(
    const channel_properties* chan,
    const float dt,
    const float dx,
    const float qup,
    const float quc,
    const float qdp,
    const float ql,
    hydraulic_geometry *hg,
    QHC *qc,
) nogil:

    compute_hydraulic_geometry(qc.h, chan, hg)
    compute_celerity(chan, hg, qc)
    qc.cn = qc.ck * (dt/dx)

    # cdef float C_dot_Q = (qc.C1 * qup) + (qc.C2 * quc) + (qc.C3 * qdp) + qc.C4
    # cdef float areasum = 1/(hg.AREA+hg.AREAC)


    cdef float tw
    if qc.h > chan.bfd:
        tw = chan.twcc
    else:
        tw = hg.twl

    cdef float Xmin = qc.Xmin
    cdef float Xtmp = 0.5 * (1 - (qc.Q_j / (2 * tw * chan.s0 * qc.ck * dx)))
    if Xtmp <= Xmin:
       qc.X = Xmin
    elif Xtmp > Xmin and Xtmp < 0.5:
       qc.X = Xtmp
    else:
       qc.X = 0.5

    cdef float tmp1, tmp2, tmp3
    qc.Km = dx/qc.ck
    if qc.ck > 0 and dt < qc.Km:
        dt2 = dt/2.0
        tmp1 = (dt - 2.0 * qc.Km * qc.X)
        tmp2 = qc.Km * (1 - qc.X)
        tmp3 = tmp1 + 2.0 * qc.Km
        qc.C1 = (dt + 2 * qc.Km * qc.X) / tmp3
        qc.C2 = tmp1 / tmp3
        qc.C3 = (tmp2 - dt2) / (tmp2 + dt2)
        qc.C4 = (2 * ql * dt) / tmp3
    else:
        qc.D = 1.5 - qc.X    # -- seconds
        qc.C1 = (qc.X + 0.5)/qc.D
        qc.C2 = (0.5 - qc.X)/qc.D
        qc.C3 = qc.C2
        qc.C4 = ql/qc.D

#     if(qc.ck > 0.0):
#         qc.Km = fmaxf(dt,dx/qc.ck)
#     else:
#         qc.Km = dt
#     qc.D = (qc.Km*(1.0 - qc.X) + dt/2.0)
#     qc.C1 =  (qc.Km*qc.X + dt/2.0)/qc.D
#     qc.C2 =  (dt/2.0 - qc.Km*qc.X)/qc.D
#     qc.C3 =  (qc.Km*(1.0-qc.X)-dt/2.0)/qc.D
#     qc.C4 =  (ql*dt)/qc.D

    cdef float t
    t = (qc.C1 * qup) + (qc.C2 * quc) + (qc.C3 * qdp)
    qc.C4 = fmaxf(-t, qc.C4) # qc.C4 cannot be more negative than the sum of other terms

    if hg.WP + hg.WPC > 0:
        qc.Q_mc = (t + qc.C4)
        qc.Q_normal = (
            (1.0 / (((hg.WP * chan.n) + (hg.WPC * chan.ncc)) / (hg.WP + hg.WPC)))
            * (hg.AREA + hg.AREAC)
            * (hg.R ** (2.0 / 3.0))
            * chan.sqrt_s0
        )
        qc.Q_j = qc.Q_mc - qc.Q_normal
    else:
        qc.Q_j = 0.0


cdef inline void compute_celerity(
    const channel_properties *chan,
    const hydraulic_geometry *hg,
    QHC *qc,
) nogil:

    # if (qc.h > 0.0):
    #     qc.ck = fmaxf(0,(chan.sqrt_s0/chan.n)*
    #         ((5.0/3.0)*hg.R**(2.0/3.0) -
    #         ((2.0/3.0)*hg.R**(5.0/3.0) *
    #         (2.0*chan.sqrt_1z2/(chan.bw+2.0*hg.h_lt_bf*chan.z)))))
    #     if (qc.h > chan.bfd):
    #         qc.ck = fmaxf(0.0,(qc.ck
    #                 * hg.AREA
    #                 + ((chan.sqrt_s0/(chan.ncc))
    #                 * (5.0/3.0)*(hg.h_gt_bf)**(2.0/3.0))
    #                 * hg.AREAC)
    #                 / (hg.AREA+hg.AREAC))
    # else:
    #     qc.ck = 0.0

    if qc.h > chan.bfd:
        qc.ck = fmaxf(
            0.0,
            (
                (chan.sqrt_s0 / chan.n)
                * (
                    (5.0 / 3.0) * hg.R ** (2.0 / 3.0)
                    - (
                        (2.0 / 3.0)
                        * hg.R ** (5.0 / 3.0)
                        * (
                            2.0
                            * chan.sqrt_1z2
                            / (chan.bw + 2.0 * chan.bfd * chan.z)
                        )
                    )
                )
                * hg.AREA
                + (
                    (chan.sqrt_s0 / chan.ncc)
                    * (5.0 / 3.0)
                    * (qc.h - chan.bfd) ** (2.0 / 3.0)
                )
                * hg.AREAC
            )
            / (hg.AREA + hg.AREAC),
        )
    else:
        if qc.h > 0.0:  # avoid divide by zero
            qc.ck = fmaxf(
                0.0,
                (chan.sqrt_s0 / chan.n)
                * (
                    (5.0 / 3.0) * hg.R ** (2.0 / 3.0)
                    - (
                        (2.0 / 3.0)
                        * hg.R ** (5.0 / 3.0)
                        * (
                            2.0
                            * chan.sqrt_1z2
                            / (chan.bw + 2.0 * qc.h * chan.z)
                        )
                    )
                ),
            )
        else:
            qc.ck = 0.0


cdef inline void compute_hydraulic_geometry(
    const float h,
    const channel_properties *chan,
    hydraulic_geometry *hg,
) nogil:

     hg.twl = chan.bw + 2*chan.z*h

     hg.h_gt_bf = fmaxf(h - chan.bfd, 0)
     hg.h_lt_bf = fminf(chan.bfd, h)

     hg.AREA = (chan.bw + hg.h_lt_bf * chan.z) * hg.h_lt_bf

     hg.WP = (chan.bw + 2 * hg.h_lt_bf * chan.sqrt_1z2)

     hg.AREAC = (chan.twcc * hg.h_gt_bf)

     if(hg.h_gt_bf > 0):
         hg.WPC = chan.twcc + (2 * (hg.h_gt_bf))
     else:
         hg.WPC = 0

     hg.R   = (hg.AREA + hg.AREAC)/(hg.WP + hg.WPC)
     # R = (h*(bw + twl) / 2.0_prec) / (bw + 2.0_prec*(((twl - bw) / 2.0_prec)**2.0_prec + h**2.0_prec)**0.5_prec)


