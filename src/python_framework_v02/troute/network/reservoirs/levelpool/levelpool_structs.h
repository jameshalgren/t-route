#ifndef LEVELPOOL_STRUCTS_H
#define LEVELPOOL_STRUCTS_H
/*
    C Structures
*/
#include "../../reach_structs.h"
typedef struct {
  int lake_number;
  double dam_length, area, max_depth;
  double orifice_area, orifice_coefficient, orifice_elevation;
  double weir_coefficient, weir_elevation, weir_length;
  double initial_fractional_depth, water_elevation;

  //Handle to operate levelpool reservoir
  void* handle;
} _MC_Levelpool;

#endif //LEVELPOOL_STRUCTS_H
