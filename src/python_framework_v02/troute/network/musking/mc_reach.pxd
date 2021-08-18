cimport numpy as np
"""
FIXME add some significant inline documentation
"""
from troute.network.reach cimport Reach, Segment

cdef extern from "mc_reach_structs.c":
    _MC_Segment get_mc_segment(_Reach* reach, int index) nogil
cdef extern from "mc_reach_structs.h":
  ctypedef struct _MC_Segment:
    long id;
    double dt, dx, bw, tw, twcc, n, ncc, cs, s0;
    double qdp, velp, depthp;
  ctypedef struct _MC_Reach:
    pass
  ctypedef struct _Reach:
    pass

cdef class MC_Segment(Segment):
  """
    A muskingcung segment
  """
  #TODO document these attributes
  cdef readonly double dt, dx, bw, tw, twcc, n, ncc, cs, s0
  cdef readonly double qdp, velp, depthp

cdef class MC_Reach(Reach):
  """
    A muskingcung reach -> collection of ordered MC_Segments
  """
  cdef _MC_Segment* _segments
  cdef int _num_segments # C only accessible
  cdef readonly num_segments #Python accessible, readonly outside class
  cdef readonly list segments
  cdef route(self)
