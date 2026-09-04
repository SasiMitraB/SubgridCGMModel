cd /home/sasi/Projects/SubgridCGMModel/builds/hr_build_mpi
cmake -S ../../athenak -B . -DCMAKE_BUILD_TYPE=Release -DPROBLEM=kh_radiative_cooling
cmake --build . -j$(nproc)
