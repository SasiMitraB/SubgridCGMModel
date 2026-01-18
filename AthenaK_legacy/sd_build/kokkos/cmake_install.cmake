# Install script for directory: /ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/kokkos

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/core/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/containers/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/algorithms/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/simd/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/example/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/benchmarks/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/cmake/Kokkos" TYPE FILE FILES
    "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/KokkosConfig.cmake"
    "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/KokkosConfigCommon.cmake"
    "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/KokkosConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/Kokkos/KokkosTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/Kokkos/KokkosTargets.cmake"
         "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/CMakeFiles/Export/18c67999278cb14d1a9023a645a204b4/KokkosTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/Kokkos/KokkosTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib64/cmake/Kokkos/KokkosTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/cmake/Kokkos" TYPE FILE FILES "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/CMakeFiles/Export/18c67999278cb14d1a9023a645a204b4/KokkosTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/cmake/Kokkos" TYPE FILE FILES "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/CMakeFiles/Export/18c67999278cb14d1a9023a645a204b4/KokkosTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kokkos" TYPE FILE FILES "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/KokkosCore_config.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM FILES
    "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/kokkos/bin/nvcc_wrapper"
    "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/kokkos/bin/hpcbind"
    "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/temp/kokkos_launch_compiler"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/kokkos" TYPE FILE FILES
    "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/KokkosCore_config.h"
    "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/KokkosCore_Config_FwdBackend.hpp"
    "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/KokkosCore_Config_SetupBackend.hpp"
    "/ptmp/mpa/dipda/subgrid/SubgridCGMModel/AthenaK_legacy/sd_build/kokkos/KokkosCore_Config_DeclareBackend.hpp"
    )
endif()

