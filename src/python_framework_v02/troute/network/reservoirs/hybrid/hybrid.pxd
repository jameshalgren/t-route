cimport numpy as np
"""
Declaring C types for Hybrid Class variables and functions
"""
from troute.network.reach cimport Reach, compute_type

############ Other Reservoir Interface ############
cdef void run_hybrid_c(_Reach* reach, double inflow, double lateral_inflow, double routing_period, double* outflow, double* water_elevation) nogil

cdef extern from "hybrid_structs.h":
  ctypedef struct _MC_Hybrid:
    int lake_number
    double dam_length, area, max_depth
    double orifice_area, orifice_coefficient, orifice_elevation
    double weir_coefficient, weir_elevation, weir_length
    double initial_fractional_depth, water_elevation
    int reservoir_type
    char* reservoir_parameter_file
    char* start_date
    char* usgs_timeslice_path
    char* usace_timeslice_path
    int observation_lookback_hours
    int observation_update_time_interval_seconds
  ctypedef struct _Reach:
    pass

cdef class MC_Hybrid(Reach):
  """
  C type for MC_Hybrid which is a reservoir subclass of a Reach
  """
  cpdef (double,double) run(self, double inflow, double lateral_inflow, double routing_period)
