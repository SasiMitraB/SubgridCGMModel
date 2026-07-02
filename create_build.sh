cd /home/sasi/Projects/SubgridCGMModel/builds
rm -rf hr_build_mpi
mkdir hr_build_mpi
cd hr_build_mpi

cmake -S ../../athenak -B . \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=kh_radiative_cooling \
  -DAthena_ENABLE_MPI=ON \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc

cmake --build . -j$(nproc)