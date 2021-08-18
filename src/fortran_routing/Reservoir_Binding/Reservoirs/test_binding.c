#include <stdio.h>

extern void* get_lp_handle();

extern void init_lp(void* handle, double *water_elevation, double *lake_area, double *weir_elevation, double *weir_coefficient, double *weir_length, double *dam_length, double *orifice_elevation, double *orifice_coefficient, double *orifice_area, double *max_depth, int *lake_number);

extern void run_lp(void* handle, double *previous_timestep_inflow, double *inflow, double *lateral_inflow, double *water_elevation, double *outflow, double *routing_period, int *dynamic_reservoir_type, double *assimilated_value, char *assimilated_source_file);

extern void free_lp(void* handle);

int main(int argc, char** argv)
{
    double water_elevation = 0.0;
    double lake_area = 1.509490013122558594e+01;
    double weir_elevation = 9.626000022888183238e+00;
    double weir_coefficient = 0.4;
    double weir_length = 1.000000000000000000e+01;
    double dam_length = 10.0;
    double orifice_elevation = 7.733333269755045869e+00;
    double orifice_coefficient = 1.000000000000000056e-01;
    double orifice_area = 1.0;
    double max_depth = 9.960000038146972656e+00;
    int lake_number = 16944276;
    void* test_p = get_lp_handle();

    init_lp(test_p, &water_elevation, &lake_area, &weir_elevation, &weir_coefficient, &weir_length, &dam_length, &orifice_elevation, &orifice_coefficient, &orifice_area, &max_depth, &lake_number);

    double previous_timestep_inflow = 0.0;
    double inflow = 0.0;
    double lateral_inflow = 0.0;
    double outflow = 0.0;
    double routing_period = 300.0;
    int dynamic_reservoir_type = 1;
    double assimilated_value = 0.0;
    char assimilated_source_file[256];
    water_elevation = 9.73733330;
    
    run_lp(test_p, &previous_timestep_inflow, &inflow, &lateral_inflow, &water_elevation, &outflow, &routing_period, &dynamic_reservoir_type, &assimilated_value, assimilated_source_file);
    
    printf ("Outflow: %f\n", outflow);
    printf("Complete \n");
}

