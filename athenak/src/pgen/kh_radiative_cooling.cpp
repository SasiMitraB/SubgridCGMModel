
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kh_radiative_cooling.cpp
//  \brief Problem generator for KH instability with radiative cooling.
//  Sets up different initial conditions selected by flag "iprob"
//    - iprob=1 : tanh profile with multiple mode perturbation
//  Can add other iprob flags for different initial setups.
//  Can also turn on radiative cooling by setting "use_radiation" to true in the input file.

#include <iostream>
#include <sstream>
#include <cstdio>
#include <vector>
#include <cmath>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "pgen.hpp"
#include "srcterms/ismcooling.hpp" // Included ISM cooling function
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
//! \fn
//  \brief Problem Generator for KHI with radiative cooling

Real pressure, density_hot, density_cold, velocityx_hot, velocityx_cold, scalar_hot, scalar_cold; // global variables for the boundary function

void constant_bcs(Mesh *pm); // forward declaration

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

  // read problem parameters from input file
  int iprob  = pin->GetReal("problem","iprob");
  std::string init_file = pin->GetOrAddString("problem", "init_file", "none");
  Real amp   = pin->GetOrAddReal("problem","amp",0.01);                   // amplitude of perturbation  
  Real sigma = pin->GetOrAddReal("problem","sigma",0.05);                 // characteristic length for the region where perturbation is applied
  Real vx_hot = pin->GetOrAddReal("problem","vx_hot",0.0);                 // x-velocity of the hot phase
  Real vx_cold = pin->GetOrAddReal("problem","vx_cold",0.0);               // x-velocity of the cold phase
  Real a_char = pin->GetOrAddReal("problem","a_char", 0.01);         // characteristic width of the interface
  Real rho_cold  = pin->GetOrAddReal("problem","rho_cold",1.0);      // cold phase density
  Real rho_hot   = pin->GetOrAddReal("problem","rho_hot",0.1);       // hot temp phase density
  Real y0    = pin->GetOrAddReal("problem","y0",0.5);                // mean scalar value
  Real y1    = pin->GetOrAddReal("problem","y1",0.5);                // difference in scalar values for both phases
  Real p_in  = pin->GetOrAddReal("problem","press",20.0);       	   // initial pressure
  Real cold_frac = pin->GetOrAddReal("problem","cold_frac",0.5);     // fraction of the domain in y-direction that is cold

  // initialising globals
  pressure = p_in;
  density_cold = rho_cold;              // density of the cold phase
  density_hot  = rho_hot;               // density of the hot phase
  velocityx_hot = vx_hot;                // x-velocity of the hot phase
  velocityx_cold = vx_cold;              // x-velocity of the cold phase
  scalar_cold = y0+y1;                  // scalar value for high density region
  scalar_hot  = y0-y1;                  // scalar value for low density region

  //user_hist_func = KHHistory;
  user_bcs_func = constant_bcs;

  if (restart) return;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  Real gm1;
  int nfluid, nscalars;                 // number of fluid variables and scalars

  if (pmbp->phydro != nullptr) {
    gm1 = (pmbp->phydro->peos->eos_data.gamma) - 1.0;
    nfluid = pmbp->phydro->nhydro;
    nscalars = pmbp->phydro->nscalars;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "This simulation requires Hydro" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &w0_ = pmbp->phydro->w0;

  if (nscalars == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "This simulation requires nscalars != 0" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Coordinates of mesh extremes
  Real x1min_mesh = pmy_mesh_->mesh_size.x1min;         
  Real x1max_mesh = pmy_mesh_->mesh_size.x1max;
  Real x2min_mesh = pmy_mesh_->mesh_size.x2min;
  Real x2max_mesh = pmy_mesh_->mesh_size.x2max;

  Real L_x = x1max_mesh - x1min_mesh;             // length of the domain in x1 direction
  Real L_y = x2max_mesh - x2min_mesh;             // length of the domain in x2 direction
  Real y_cold = x2min_mesh + cold_frac*(L_y);     // y position of the interface between the two phases
  Real rho0  = (rho_cold + rho_hot)/2;            // density mean
  Real rho1 = (rho_cold - rho_hot)/2;             // density difference/2
  Real vshear_half = (vx_hot + vx_cold)/2;
  Real vshear_delta = (vx_hot - vx_cold)/2;

  units::Units my_unit(pin);

  // initialize primitive variables
  if (iprob == 2 || init_file != "none") {
    int Nx1_mesh = pmy_mesh_->mesh_indcs.nx1;
    int Nx2_mesh = pmy_mesh_->mesh_indcs.nx2;
    const int num_vars = 7;

    std::vector<double> host_data(num_vars * Nx2_mesh * Nx1_mesh);
    FILE *fp = std::fopen(init_file.c_str(), "rb");
    if (!fp) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Cannot open initial data file: " << init_file << std::endl;
      std::exit(EXIT_FAILURE);
    }
    size_t read_count = std::fread(host_data.data(), sizeof(double), host_data.size(), fp);
    if (read_count != host_data.size()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Failed to read expected data elements (" << read_count << " / " << host_data.size()
                << ") from: " << init_file << std::endl;
      std::fclose(fp);
      std::exit(EXIT_FAILURE);
    }
    std::fclose(fp);

    DvceArray3D<Real> d_ic_data("d_ic_data", num_vars, Nx2_mesh, Nx1_mesh);
    auto h_ic_data = Kokkos::create_mirror(d_ic_data);

    for (int v = 0; v < num_vars; ++v) {
      for (int j = 0; j < Nx2_mesh; ++j) {
        for (int i = 0; i < Nx1_mesh; ++i) {
          h_ic_data(v, j, i) = host_data[v * (Nx2_mesh * Nx1_mesh) + j * Nx1_mesh + i];
        }
      }
    }
    Kokkos::deep_copy(d_ic_data, h_ic_data);

    Real dx1 = pmy_mesh_->mesh_size.dx1;
    Real dx2 = pmy_mesh_->mesh_size.dx2;

    par_for("KHI_FromFile", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j - js, nx2, x2min, x2max);

      int ig = static_cast<int>((x1v - x1min_mesh) / dx1);
      int jg = static_cast<int>((x2v - x2min_mesh) / dx2);
      ig = (ig < 0) ? 0 : ((ig >= Nx1_mesh) ? (Nx1_mesh - 1) : ig);
      jg = (jg < 0) ? 0 : ((jg >= Nx2_mesh) ? (Nx2_mesh - 1) : jg);

      Real dens  = d_ic_data(0, jg, ig);
      Real eint  = d_ic_data(1, jg, ig);
      Real vx    = d_ic_data(2, jg, ig);
      Real vy    = d_ic_data(3, jg, ig);
      Real vz    = d_ic_data(4, jg, ig);
      Real scal1 = d_ic_data(5, jg, ig);
      Real scal2 = d_ic_data(6, jg, ig);

      w0_(m, IDN, k, j, i) = dens;
      w0_(m, IEN, k, j, i) = eint;
      w0_(m, IVX, k, j, i) = vx;
      w0_(m, IVY, k, j, i) = vy;
      w0_(m, IVZ, k, j, i) = vz;
      w0_(m, nfluid, k, j, i) = scal1;
      if (nscalars > 1) {
        w0_(m, nfluid + 1, k, j, i) = scal2;
      }
      for (int n = nfluid + 2; n < (nfluid + nscalars); ++n) {
        w0_(m, n, k, j, i) = 0.0;
      }
    });
  } else {
    par_for("KHI", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {

      // Calculating the cell center coordinates
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real dens,pres,vx,vy,vz,scal;

      if (iprob == 1) {
          pres = p_in;
          dens = rho0 - rho1*tanh((x2v - y_cold)/a_char);
          vx = vshear_half + vshear_delta*tanh((x2v - y_cold)/a_char);            // this makes relative shear velocity = vx_hot - vx_cold.
          // Adding perturbations to vy. The perturbation is a sum of sine functions with different wavelengths.
          // wavenumbers are k_n = 2n*pi/L_x, where n = 5,10,18,25,32.
          Real perturb = sin(2.0*5.0*M_PI*x1v/L_x)+sin(2.0*10.0*M_PI*x1v/L_x)+sin(2.0*18.0*M_PI*x1v/L_x)+sin(2.0*25.0*M_PI*x1v/L_x)+sin(2.0*32.0*M_PI*x1v/L_x);
          vy = -amp*2.0*vshear_delta*(perturb)*exp( -SQR((x2v - y_cold)/sigma) );
          vz = 0.0;
          scal = y0 - y1*tanh((x2v - y_cold)/a_char);
      }

      // setting primitives
      w0_(m,IDN,k,j,i) = dens;
      w0_(m,IEN,k,j,i) = pres/gm1;
      w0_(m,IVX,k,j,i) = vx;
      w0_(m,IVY,k,j,i) = vy;
      w0_(m,IVZ,k,j,i) = vz;
      // adding passive scalars
      for (int n=nfluid; n<(nfluid+nscalars); ++n) {
        w0_(m,n,k,j,i) = scal;
      } 
    });
  }

  // Convert primitives to conserved
  if (pmbp->phydro != nullptr) {
      auto &u0_ = pmbp->phydro->u0;
      pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
    } 

  return;
}

void constant_bcs (Mesh *pm) {
    auto &indcs = pm->mb_indcs;
    int &ng = indcs.ng;
    int n1 = indcs.nx1 + 2*ng;
    int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
    int &is = indcs.is;  int &ie  = indcs.ie;
    int &js = indcs.js;  int &je  = indcs.je;
    int &ks = indcs.ks;  int &ke  = indcs.ke;
    auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
    MeshBlockPack *pmbp = pm->pmb_pack;
  
    Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  
    DvceArray5D<Real> u0_, w0_;
    u0_ = pm->pmb_pack->phydro->u0;
    w0_ = pm->pmb_pack->phydro->w0;
    int nmb = pm->pmb_pack->nmb_thispack;
    int &nfluid = pmbp->phydro->nhydro;
    int &nscalars = pmbp->phydro->nscalars;

    // ---- device copies of global variables (needed for GPU kernels) ----
    const Real pressure_d       = pressure;
    const Real density_hot_d    = density_hot;
    const Real density_cold_d   = density_cold;
    const Real velocityx_hot_d  = velocityx_hot;
    const Real velocityx_cold_d = velocityx_cold;
    const Real scalar_hot_d     = scalar_hot;
    const Real scalar_cold_d    = scalar_cold;

    // ConsToPrim over all X2 ghost zones *and* at the innermost/outermost X2-active zones
    // of Meshblocks, even if Meshblock face is not at the edge of computational domain
    if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js,0,(n3-1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je,je+ng,0,(n3-1));
    }
    
    par_for("kh_bcs", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
      KOKKOS_LAMBDA(int m, int k, int i) {
        if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          w0_(m,IDN,k,js-j-1,i) = density_cold_d;
          w0_(m,IEN,k,js-j-1,i) = pressure_d/gm1;
          w0_(m,IVX,k,js-j-1,i) = velocityx_cold_d;
          w0_(m,IVY,k,js-j-1,i) = w0_(m,IVY,k,js,i); //outflow
          //w0_(m,IVY,k,js-j-1,i) = -w0_(m,IVY,k,js,i)*w0_(m,IDN,k,js,i)/density_cold; //reflective, better mass conservation
          //w0_(m,IVY,k,js-j-1,i) = -w0_(m,IVY,k,js,i);
          w0_(m,IVZ,k,js-j-1,i) = w0_(m,IVZ,k,js,i);
          for (int n=nfluid; n<(nfluid+nscalars); ++n) {
            w0_(m,n,k,js-j-1,i) = scalar_cold_d;
          }
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2)==BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          w0_(m,IDN,k,je+j+1,i) = density_hot_d;
          w0_(m,IEN,k,je+j+1,i) = pressure_d/gm1;
          w0_(m,IVX,k,je+j+1,i) = velocityx_hot_d;
          w0_(m,IVY,k,je+j+1,i) = w0_(m,IVY,k,je,i); //outflow
          //w0_(m,IVY,k,je+j+1,i) = -w0_(m,IVY,k,je,i)*w0_(m,IDN,k,je,i)/density_hot; //reflective, better mass conservation
          //w0_(m,IVY,k,je+j+1,i) = -w0_(m,IVY,k,je,i);
          w0_(m,IVZ,k,je+j+1,i) = w0_(m,IVZ,k,je,i);
          for (int n=nfluid; n<(nfluid+nscalars); ++n) {
            w0_(m,n,k,je+j+1,i) = scalar_hot_d;
          }
        }
      }
    });
    
    // Convert primitives to conserved
    // PrimToCons on X2 ghost zones
    if (pm->pmb_pack->phydro != nullptr) {
      pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
      pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
    } 
  }