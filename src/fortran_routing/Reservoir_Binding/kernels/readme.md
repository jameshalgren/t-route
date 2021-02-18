The Reservoir Module Requires NetCDF C and Fortran Libraries.
On a Linux system, the following environment variables should
be set if these libraries exist on that system:
1. Type "export NETCDF_LIB=/usr/lib64/openmpi/lib/"
2. Type "export NETCDF_INC=/usr/include/openmpi-x86_64"


In order to set up the tests for all reservoir types:
1. Activate a python3 virtual environment
2. Type "pip install Cython"
3. Type "pip install pytest"
4. Type "pip install pytest-parallel"
5. Navigate to "fortran/Reservoirs" directory
6. Type "make" which builds the Fortran Reservoir Module
7. Navigate back to "kernels" directory
8. Type "python setup.py install" which compiles and links the Cython handles to
   each Reservoir type


Run the following tests:
1. Type "pytest" which runs 4 reservoirs in one process in one thread sequentially
2. Type "pytest --workers 4" which runs 4 reservoirs in parallel in 4 separate processes,
   and each process runs a single thread
3. Type "pytest --tests-per-worker 4" which runs 4 reservoirs in 1 process on 4 parallel threads

You should see all 4 tests passed for each of these 3 runs 


