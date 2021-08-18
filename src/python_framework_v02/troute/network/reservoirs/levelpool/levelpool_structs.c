#include <stdlib.h>
#include "../../reach_structs.h"
#include <stdio.h>
/* Level Pool Reservoir Interface */
extern void* get_lp_handle();

extern void init_lp(void* handle, double *water_elevation, double *lake_area, double *weir_elevation,
                    double *weir_coefficient, double *weir_length, double *dam_length, double *orifice_elevation,
                    double *orifice_coefficient, double *orifice_area, double *max_depth, int *lake_number);

extern void run_lp(void* handle, double *inflow, double *lateral_inflow,
                    double *water_elevation, double *outflow, double *routing_period);

extern void free_lp(void* handle);

init_levelpool_reach(_Reach* reach, int lake_number,
                          double dam_length, double area, double max_depth,
                          double orifice_area, double orifice_coefficient, double orifice_elevation,
                          double weir_coefficient, double weir_elevation, double weir_length,
                          double initial_fractional_depth, double water_elevation
)
{
  if( reach != NULL )
  {
    reach->reach.lp.lake_number = lake_number;
    reach->reach.lp.dam_length = dam_length;
    reach->reach.lp.area = area;
    reach->reach.lp.max_depth = max_depth;
    reach->reach.lp.orifice_area = orifice_area;
    reach->reach.lp.orifice_coefficient = orifice_coefficient;
    reach->reach.lp.orifice_elevation = orifice_elevation;
    reach->reach.lp.weir_coefficient = weir_coefficient;
    reach->reach.lp.weir_elevation = weir_elevation;
    reach->reach.lp.weir_length = weir_length;
    reach->reach.lp.initial_fractional_depth = initial_fractional_depth;

    if(water_elevation < 0){
      //Equation below is used in wrf-hydro
      printf("WARNING: LEVELPOOL USING COLDSTART WATER ELEVATION\n");
      fflush(stdout);
      reach->reach.lp.water_elevation = orifice_elevation + ((max_depth - orifice_elevation) * initial_fractional_depth);
    }
    else{
      reach->reach.lp.water_elevation = water_elevation;
    }

    reach->reach.lp.handle = get_lp_handle();
    init_lp(reach->reach.lp.handle, &reach->reach.lp.water_elevation, &reach->reach.lp.area,
                 &reach->reach.lp.weir_elevation, &reach->reach.lp.weir_coefficient, &reach->reach.lp.weir_length,
                 &reach->reach.lp.dam_length, &reach->reach.lp.orifice_elevation, &reach->reach.lp.orifice_coefficient,
                 &reach->reach.lp.orifice_area, &reach->reach.lp.max_depth, &reach->reach.lp.lake_number);
  }
}

void free_levelpool_reach(_Reach* reach)
{
  free_lp(reach->reach.lp.handle);
}

void route(_Reach* reach, double inflow, double lateral_inflow, double routing_period,
           double* outflow, double* water_elevation)
{
  run_lp(reach->reach.lp.handle, &inflow, &lateral_inflow, &reach->reach.lp.water_elevation, outflow, &routing_period);
  *water_elevation = reach->reach.lp.water_elevation;
}
