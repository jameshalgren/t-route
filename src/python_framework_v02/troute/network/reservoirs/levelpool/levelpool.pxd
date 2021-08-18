cimport numpy as np
"""
Declaring C types for Level Pool Class variables and functions
"""
from troute.network.reach cimport Reach, compute_type

############ Other Reservoir Interface ############
cdef void run_lp_c(_Reach* reach, double inflow, double lateral_inflow, double routing_period, double* outflow,  double* water_elevation) nogil

cdef extern from "levelpool_structs.h":
  ctypedef struct _MC_Levelpool:
    int lake_number
    double dam_length, area, max_depth
    double orifice_area, orifice_coefficient, orifice_elevation
    double weir_coefficient, weir_elevation, weir_length
    double initial_fractional_depth, water_elevation
  ctypedef struct _Reach:
    pass

cdef class MC_Levelpool(Reach):
  """
  C type for MC_Levelpool which is a resevoir subclass of a Reach
  """
  cpdef (double,double) run(self, double inflow, double lateral_inflow, double routing_period)
