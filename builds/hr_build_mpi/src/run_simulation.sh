cd /home/sasi/Projects/SubgridCGMModel/builds/hr_build_mpi/src

# Run with N MPI ranks (e.g., 4). Adjust to your machine/input.
mpirun -np ${NPROCS:-16} ./athena \
  -i "${1:-kh_radiative_512.athinput}" \
  -d /home/sasi/Projects/SubgridCGMModel/simulation_outputs/hr_build_512
