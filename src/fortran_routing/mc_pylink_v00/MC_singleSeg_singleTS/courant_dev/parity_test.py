# -*- coding: utf-8 -*-
"""parity_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/NOAA-OWP/t-route/blob/264af716f5261351b0e976c808ba731d04782a4c/src/fortran_routing/mc_pylink_v00/MC_singleSeg_singleTS/courant_dev/parity_test.ipynb

# Testing Courant capability in Muskingum-Cunge FORTRAN module
The Courant number is an important diagnostic metric for river routing models, defined as the ratio of the kinematic wave celerity to numerical celerity (the grid ratio Î”x/Î”t). According to Ponce, the Muskingum-Cunge routing method is a reasonable representation of the physical prototype if the Courant number is close to 1 (among other parametric limitations). Therefore, to assess the numerical accuracy of Musking-Cunge routing simulations across the NHDPlus, it is adventageous to  output the Courant condition for each segment and timestep of a routing simulation. 

Here, we test an additional subroutine added to the Muskingum-Cunge FORTRAN module to calculate Courant condition and kinematic celerity upon every call of the model. Additionally the MC module is updated to return the X parameter, another valuable diagnostic. This parity tests for any unintended changes in simulated flow, velocity, and depth brought about by the newly implemented subroutine. Additionally, the Courant number and kinematic celerity values calculated by the new subroutine are compared against Courant number and kinematic celerity results calculated with a simple Python function, using outputs from the original MC module. 

A few important notes before we dive in... 
1. The modifications we are testing on top of `t-route/src/fortran_routing/mc_pylink_v00/MC_singleSeg_singleTS/MCsingleSegStime_f2py_NOLOOP.f90`.  This file contains a new subroutine to calculate and output the Courant number and kinematic celerity.
2. `t-route/src/fortran_routing/mc_pylink_v00/MC_singleSeg_singleTS/MUSKINGCUNGE.f90` is an unaltered base case module for comparison.
"""

import sys
import os
import subprocess
import csv
import numpy as np
import time

try:
    import google.colab

    ENV_IS_CL = True
    root = r"/content/t-route"
    
    # TO DO: revise this to clone master prior to merge. 
    subprocess.run(["git", 
                    "clone", 
                    "-b",
                    "courant",
                    "https://github.com/awlostowski-noaa/t-route.git"])
    
    
except:
    root = os.path.dirname(os.path.abspath("../../../../"))

# development directory contains unmodified, or "base", MC FORTRAN module
fortran_routing_directory = os.path.join(root,"src","fortran_routing","mc_pylink_v00","MC_singleSeg_singleTS")
sys.path.append(fortran_routing_directory)

v01_routing_directory = os.path.join(root,"src","python_routing_v01")
sys.path.append(v01_routing_directory)  

v01_framework_directory = os.path.join(root,"src","python_framework_v01") 
sys.path.append(v01_framework_directory) 

v02_framework_directory = os.path.join(root,"src","python_framework_v02") 
sys.path.append(v02_framework_directory)

"""# Specify model parameters
The parameters needed to run the Muskingum-Cunge routing model are contained in a Python dictionary object. We will pass the same parameters to base and development versions of the MC module and expect identical results.
"""

# model parameters
params = {}
params["dt"] = 60.0
params["dx"] = 1800.0
params["bw"] = 112.0
params["tw"] = 248.0
params["twcc"] = 623.60
params["n"] = 0.02800000086426735
params["ncc"] = 0.03136000037193298
params["cs"] = 0.42
params["s0"] = 0.007999999690800905
params["qlat"] = 40.0
params["qup"] = 45009.0
params["quc"] = 50098.0
params["qdp"] = 50014.0
params["depthp"] = 30.0

"""# Compile and run the edited module"""

# Compile MC module
f2py_call = []
f2py_call.append(r"f2py3")
f2py_call.append(r"-c")
f2py_call.append(r"varPrecision.f90")
f2py_call.append(r"MCsingleSegStime_f2py_NOLOOP.f90")
f2py_call.append(r"-m")
f2py_call.append(r"mc_sseg_stime")
f2py_call.append(r"--opt='-O3'")
# f2py_call.append(r"--noopt")

subprocess.run(
    f2py_call,
    cwd=fortran_routing_directory,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# import Python shared object 
try:
    # import MC subroutine from shared object
    from mc_sseg_stime import muskingcunge_module as mc

except:
    print("ERROR: Couldn't find mc_sseg_stime shared_anw object. There may be a problem with the f2py compile process.")

# run the model
qdc, velc, depthc, ck, cn, x = mc.muskingcungenwm(
#qdc, velc, depthc = mc.muskingcungenwm(
            params["dt"],
            params["qup"],
            params["quc"],
            params["qdp"],
            params["qlat"],
            params["dx"],
            params["bw"],
            params["tw"],
            params["twcc"],
            params["n"],
            params["ncc"],
            params["cs"],
            params["s0"],
            0,
            params["depthp"]
        )

# cache output
dev_result = {}
dev_result["qdc"] = qdc
dev_result["velc"] = velc
dev_result["depthc"] = depthc
dev_result["ck"] = ck
dev_result["cn"] = cn

"""# Compile and run the original module"""

# Compile MC module
f2py_call = []
f2py_call.append(r"f2py3")
f2py_call.append(r"-c")
f2py_call.append(r"varPrecision.f90")
f2py_call.append(r"MUSKINGCUNGE.f90")
f2py_call.append(r"-m")
f2py_call.append(r"mc_wrf_hydro")
f2py_call.append(r"--opt='-O3'")

subprocess.run(
    f2py_call,
    cwd=fortran_routing_directory,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# import Python shared object 
try:
    # import MC subroutine from shared object
    from mc_wrf_hydro import submuskingcunge_wrf_module as mc_og

except:
    print("ERROR: Couldn't find mc_sseg_stime shared object. There may be a problem with the f2py compile process.")

# run the model
qdc, velc, depthc, ck, cn, x = mc_og.submuskingcunge(
#qdc, velc, depthc = mc_og.submuskingcunge(
            params["qup"],
            params["quc"],
            params["qdp"],
            params["qlat"],
            params["dt"],
            params["s0"],
            params["dx"],
            params["n"],
            params["cs"],
            params["bw"],
            params["tw"],
            params["twcc"],
            params["ncc"],
            params["depthp"]
        )

# cache output
base_result = {}
base_result["qdc"] = qdc
base_result["velc"] = velc
base_result["depthc"] = depthc
base_result["ck"] = ck
base_result["cn"] = cn

"""# Compute Courant condition and kinematic celerity from original module results"""

# calculate kinematic celerity and courant number from model parameters and output from f2py process
def courant(depthc, cs, tw, bw, twcc, s0, n, ncc, dt, dx):
    
    # channel side distance
    z = 1.0/cs

    # bankfull depth (assumes tw > bw)
    bfd =  (tw - bw)/(2*z)
    
    if depthc > bfd:

        # ******* when depth is above bank full *******
        h = depthc

        AREA =  (bw + bfd * z) * bfd
        AREAC = (twcc * (h - bfd)) 
        WP = (bw + 2 * bfd * np.sqrt(1 + z*z))
        WPC = twcc + (2 * (h - bfd))
        R   = (AREA + AREAC)/(WP + WPC)

        ck =  ((np.sqrt(s0)/n)*((5/3)*R**(2/3) - \
                    ((2/3)*R**(5/3)*(2*np.sqrt(1 + z*z)/(bw+2*bfd*z))))*AREA \
                    + ((np.sqrt(s0)/(ncc))*(5/3)*(h - bfd)**(2/3))*AREAC)/(AREA+AREAC)
        
    else:

        # ******* when depth is below bank full ********
        h = depthc
        
        AREA = (bw + h * z ) * h
        WP = (bw + 2 * h * np.sqrt(1 + z*z))
        R = AREA / WP

        ck = (np.sqrt(s0)/n)* \
                        ((5/3)*R**(2/3)-((2/3)*R**(5/3)* \
                         (2*np.sqrt(1 + z*z)/(bw+2*h*z))))


    # Courant number
    cn = ck * (dt/dx)
    
    return ck, cn

#base_result["ck"], base_result["cn"] = courant(base_result["depthc"],
#                                                params["cs"],
#                                                params["tw"],
#                                                params["bw"],
#                                                params["twcc"],
#                                                params["s0"],
#                                                params["n"],
#                                                params["ncc"],
#                                                params["dt"],
#                                                params["dx"])

"""# Compare results between edited and original modules"""

print("Percent difference between original and updated MC module results")
print("-----------------------------------------------------------------")
for key in base_result.keys():
    pct_diff = (base_result[key] - dev_result[key])/base_result[key]
    print(key + ": " + str(pct_diff) + "%")

"""# Conclusion
1. The additional Courant subroutine did not change core simulation results; flow, velocity, and depth. 
2. Extremely minor differences in kinematic celerity and Courant number are observed between development and base modules. These difference likely stem from differences in variable precision between Python (used for calculations on base results) and FOTRAN (used for calculation development results).

# Timing test
"""

base_duration = []
dev_duration = []
for n in range(0,10):
    
    start = time.time()
    for i in range(0,100000):

        result = mc_og.submuskingcunge(
                params["qup"],
                params["quc"],
                params["qdp"],
                params["qlat"],
                params["dt"],
                params["s0"],
                params["dx"],
                params["n"],
                params["cs"],
                params["bw"],
                params["tw"],
                params["twcc"],
                params["ncc"],
                params["depthp"]
            )

    end = time.time()
    base_duration.append(end - start) # seconds

    start_dev = time.time()
    for i in range(0,100000):

        result = mc.muskingcungenwm(
                params["dt"],
                params["qup"],
                params["quc"],
                params["qdp"],
                params["qlat"],
                params["dx"],
                params["bw"],
                params["tw"],
                params["twcc"],
                params["n"],
                params["ncc"],
                params["cs"],
                params["s0"],
                0,
                params["depthp"]
            )
    end_dev = time.time()
    dev_duration.append(end_dev - start_dev) # seconds
    
    print("Completed timing test", n)

del_timing = (np.mean(dev_duration) - np.mean(base_duration))/np.mean(base_duration)

print("The muskingcunge module improvements changed the wall clock time by", round(del_timing,2)*100, "% from the base module")


