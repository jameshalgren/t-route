cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

from troute.network.reach cimport compute_type, _Reach
"""
Externally defined symbols
"""

cdef extern from "levelpool_structs.c":
  void init_levelpool_reach(_Reach* reach, int lake_number,
                            double dam_length, double area, double max_depth,
                            double orifice_area, double orifice_coefficient, double orifice_elevation,
                            double weir_coefficient, double weir_elevation, double weir_length,
                            double initial_fractional_depth, double water_elevation
  )
  void free_levelpool_reach(_Reach* reach)

  void route(_Reach* reach, double routing_period, double inflow, double lateral_inflow, double* outflow,  double* water_elevation) nogil

cdef void run_lp_c(_Reach* reach, double inflow, double lateral_inflow, double routing_period, double* outflow,  double* water_elevation) nogil:
    route(reach, inflow, lateral_inflow, routing_period, outflow, water_elevation)

cdef class MC_Levelpool(Reach):
  """
    MC_Reservoir is a subclass of MC_Reach_Base_Class
  """

  def __init__(self, long id, int lake_number, long[::1] upstream_ids, args):
    """
      Construct the kernel based on passed parameters,
      which only constructs the parent class

      Params:
        id: long
          unique identity of the reach this reservoir represents
        lake_number: int TODO (long?)
          WRF_Hydro lake number of this reservoir
        upstream_ids: array[long]
          buffer/array of upstream identifiers which contribute flow to this reservoir
        args: list
          the levelpool parameters ordered as follows:
            area = args[0]
            max_depth = args[1]
            orifice_area = args[2]
            orifice_coefficient = args[3]
            orifice_elevation  =  args[4]
            weir_coefficient = args[5]
            weir_elevation = args[6]
            weir_length = args[7]
            initial_fractional_depth  = args[8]
            water_elevation = args[10]
    """
    super().__init__(id, upstream_ids, compute_type.RESERVOIR_LP)
    # Note Some issues with __calloc__:
    # The python type isn't guaranteed to be properly constructed, so cannot depend on super class being constructured.
    # Thus I don't think we can put these C init functions in __calloc__, at least not in all cases.
    # init the backing struct, pass a dam_length of 10.0 for now

    #Setting default dam_length to 10
    dam_length = 10.0
    area = args[0]
    max_depth = args[1]
    orifice_area = args[2]
    orifice_coefficient = args[3]
    orifice_elevation = args[4]
    weir_coefficient = args[5]
    weir_elevation = args[6]
    weir_length = args[7]
    initial_fractional_depth = args[8]
    water_elevation = args[10]

    init_levelpool_reach(&self._reach, lake_number,
                         dam_length, area, max_depth,
                         orifice_area, orifice_coefficient, orifice_elevation,
                         weir_coefficient, weir_elevation, weir_length,
                         initial_fractional_depth, water_elevation)

  def __dealloc__(self):
    """
      Release pointers and resources used to construct a levelpool reach
    """
    free_levelpool_reach(&self._reach)

  cpdef (double,double) run(self, double inflow, double lateral_inflow, double routing_period):
    """
      Run the levelpool routing function

      Params:
        inflow: double
          inflow into the reservoir
        lateral_inflow: double
          lateral flows into the reservoir
        routing_period: double
          amount of time to simulatie reservoir operation for, outflow if valid until this time

      Return:
        outflow: double
          flow rate out of the reservoir valid for routing_period seconds
        water_elevation:
          reservoir water surface elevation after routing_period seconds
    """
    cdef double outflow = 0.0
    cdef double water_elevation = 0.0
    with nogil:
      route(&self._reach, inflow, lateral_inflow, routing_period, &outflow, &water_elevation)
      #printf("outflow: %f\n", outflow)
      return outflow, water_elevation

  @property
  def water_elevation(self):
    """
      Reservoir water surface elevation
    """
    return self._reach.reach.lp.water_elevation

  @property
  def lake_area(self):
    """
      Surface area of the reservoir
    """
    return self._reach.reach.lp.area

  @property
  def weir_elevation(self):
    """
      Elevation, in meters, of the bottom of the weir
    """
    return self._reach.reach.lp.weir_elevation

  @property
  def weir_coefficient(self):
    """
      Weir coefficient
    """
    return self._reach.reach.lp.weir_coefficient

  @property
  def weir_length(self):
    """
      Length of the weir, in meters
    """
    return self._reach.reach.lp.weir_length

  @property
  def dam_length(self):
    """
      Length of the dam, in meters
    """
    return self._reach.reach.lp.dam_length

  @property
  def orifice_elevation(self):
    """
      Elevation, in meters, of the orifice flow component
    """
    return self._reach.reach.lp.orifice_elevation

  @property
  def orifice_area(self):
    """
      Area of the orifice flow component, in square meters
    """
    return self._reach.reach.lp.orifice_area

  @property
  def max_depth(self):
    """
      Maximum water elevaiton, in meters, before overflow occurs
    """
    return self._reach.reach.lp.max_depth

  @property
  def lake_number(self):
    """
      WRF Hydro lake identifier
    """
    return self._reach.reach.lp.lake_number

  @property
  def initial_fractional_depth(self):
    """
      Initial water surface elevation, as a percentage of total capacity,
      to use if initial water elevation is unknown.
    """
    return self._reach.reach.lp.initial_fractional_depth
