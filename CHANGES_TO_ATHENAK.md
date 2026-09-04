# Changes to AthenaK

This document records the upstream AthenaK base version used in this repository, retrieval details, and the local modifications and additions introduced.

---

## Base AthenaK Repository Information

- **Repository URL:** `git@github.com:IAS-Astrophysics/athenak.git` (https://github.com/IAS-Astrophysics/athenak)
- **Branch:** `main`
- **Retrieval Date:** September 3, 2026
- **Base Git Commit:** [`ac306752989bc2f87c9029b2aaf453689406bdfc`](https://github.com/IAS-Astrophysics/athenak/commit/ac306752989bc2f87c9029b2aaf453689406bdfc)
- **Base Commit Message:**
  ```text
  Import Drift-Control (DC) from THC/GR-Athena++ (#770)

  * Add the THC drift control
  * Adjust drift control input file
  * Fix MPI binary header mismatch caused by driftcontrol file
  * Add DOB driftcontrol
  * Add ramp-down of driftcontrol instead of turn-off
  * Add per-axis gain for driftcontrol
  * Add bounded DOB
  ```

---

## Summary of Changes

### 1. Build System & Dependency Integration (pybind11)
- **Root CMake Configuration ([`athenak/CMakeLists.txt`](file:///home/sasi/Projects/SubgridCGMModel/athenak/CMakeLists.txt)):**
  - Added `add_subdirectory(external/pybind11)` to include pybind11 in the build.
- **Source CMake Configuration ([`athenak/src/CMakeLists.txt`](file:///home/sasi/Projects/SubgridCGMModel/athenak/src/CMakeLists.txt)):**
  - Linked `pybind11::embed` to the Athena executable (`target_link_libraries(athena PRIVATE pybind11::embed)`).
  - Added preprocessor definition `-DPYBIND` for embedding support.
- **External Dependencies ([`athenak/external/pybind11`](file:///home/sasi/Projects/SubgridCGMModel/athenak/external/pybind11)):**
  - Added pybind11 submodule/tree under `external/pybind11`.

---

### 2. Physics & Source Terms
- **Cooling Function ([`athenak/src/srcterms/ismcooling.hpp`](file:///home/sasi/Projects/SubgridCGMModel/athenak/src/srcterms/ismcooling.hpp)):**
  - Added temperature cutoffs in `ISMCoolFn(Real temp)` to turn off cooling at extreme temperatures:
    ```cpp
    // turn off cooling below in the extreme ends; Not there in the default implementation
    if (logt <= log10(1.05e4) || logt > log10(0.95e6)) {
      return 0.0;
    }
    ```

---

### 3. Problem Generators (New Files)
- **[`athenak/src/pgen/subgrid.cpp`](file:///home/sasi/Projects/SubgridCGMModel/athenak/src/pgen/subgrid.cpp):**
  - Added problem generator integrating embedded Python via `pybind11` (`pybind11/embed.h`, `pybind11/numpy.h`) alongside ISM cooling routines for subgrid modeling.
- **[`athenak/src/pgen/kh_radiative_cooling.cpp`](file:///home/sasi/Projects/SubgridCGMModel/athenak/src/pgen/kh_radiative_cooling.cpp):**
  - Added problem generator for Kelvin-Helmholtz instability simulations with radiative cooling support.
