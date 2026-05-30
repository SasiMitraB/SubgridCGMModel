cd /Volumes/PortableSSD/Projects/SubgridCGMModel/builds/hr_build

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel

cmake -S ../../athenak -B . -DCMAKE_BUILD_TYPE=Release -DPROBLEM=kh_radiative_cooling
cmake --build . -j$(nproc)

dot_clean -m /Volumes/PortableSSD/Projects/SubgridCGMModel

