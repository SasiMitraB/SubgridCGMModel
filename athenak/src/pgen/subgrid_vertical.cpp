#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "pgen.hpp"
#include "srcterms/ismcooling.hpp" // Included ISM cooling function
#include "units/units.hpp"

// pybind11 headers
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace py::literals;

//----------------------------------------------------------------------------------------
//! \fn
//  \brief Problem Generator for KHI with radiative cooling with the subgrid
//  model (vertical setup)

Real pressure, density_hot, density_cold, velocityy_hot, velocityy_cold,
    scalar_hot, scalar_cold; // global variables for the boundary function

void constant_bcs(Mesh *pm); // forward declaration
namespace {
void UserSourceTerm(Mesh *pm, const Real bdt);
void SubgridFinalize(ParameterInput *pin, Mesh *pm);
}

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  user_srcs_func = UserSourceTerm;
  user_bcs_func = constant_bcs;
  pgen_final_func = SubgridFinalize;
  // user_hist_func = KHHistory;

  // read problem parameters from input file
  int iprob = pin->GetReal("problem", "iprob");
  Real amp =
      pin->GetOrAddReal("problem", "amp", 0.01); // amplitude of perturbation
  Real sigma = pin->GetOrAddReal("problem", "sigma",
                                 0.05); // characteristic length for the region
                                        // where perturbation is applied
  Real vx_hot = pin->GetOrAddReal("problem", "vx_hot",
                                  0.0); // x-velocity of the hot phase
  Real vx_cold = pin->GetOrAddReal("problem", "vx_cold",
                                   0.0); // x-velocity of the cold phase
  Real a_char = pin->GetOrAddReal(
      "problem", "a_char", 0.01); // characteristic width of the interface
  Real rho_cold =
      pin->GetOrAddReal("problem", "rho_cold", 1.0); // cold phase density
  Real rho_hot =
      pin->GetOrAddReal("problem", "rho_hot", 0.1);  // hot temp phase density
  Real y0 = pin->GetOrAddReal("problem", "y0", 0.5); // mean scalar value
  Real y1 = pin->GetOrAddReal(
      "problem", "y1", 0.5); // difference in scalar values for both phases
  Real p_in = pin->GetOrAddReal("problem", "press", 20.0); // initial pressure
  Real cold_frac = pin->GetOrAddReal(
      "problem", "cold_frac",
      0.5); // fraction of the domain in y-direction that is cold

  // initialising globals
  pressure = p_in;
  density_cold = rho_cold;  // density of the cold phase
  density_hot = rho_hot;    // density of the hot phase
  velocityy_hot = vx_hot;   // y-velocity of the hot phase (using vx_hot input param)
  velocityy_cold = vx_cold; // y-velocity of the cold phase (using vx_cold input param)
  scalar_cold = y0 + y1;    // scalar value for high density region
  scalar_hot = y0 - y1;     // scalar value for low density region

  if (restart)
    return;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  Real gm1;
  int nfluid, nscalars; // number of fluid variables and scalars

  if (pmbp->phydro != nullptr) {
    gm1 = (pmbp->phydro->peos->eos_data.gamma) - 1.0;
    nfluid = pmbp->phydro->nhydro;
    nscalars = pmbp->phydro->nscalars;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "This simulation requires Hydro" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &w0_ = pmbp->phydro->w0;

  if (nscalars == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "This simulation requires nscalars != 0" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Coordinates of mesh extremes
  Real x1min_mesh = pmy_mesh_->mesh_size.x1min;
  Real x1max_mesh = pmy_mesh_->mesh_size.x1max;
  Real x2min_mesh = pmy_mesh_->mesh_size.x2min;
  Real x2max_mesh = pmy_mesh_->mesh_size.x2max;

  Real L_x = x1max_mesh - x1min_mesh; // length of the domain in x1 direction
  Real L_y = x2max_mesh - x2min_mesh; // length of the domain in x2 direction
  Real x_cold =
      x1min_mesh +
      cold_frac * (L_x); // x position of the interface between the two phases
  Real rho0 = (rho_cold + rho_hot) / 2; // density mean
  Real rho1 = (rho_cold - rho_hot) / 2; // density difference/2
  Real vshear_half = (vx_hot + vx_cold) / 2;
  Real vshear_delta = (vx_hot - vx_cold) / 2;

  units::Units my_unit(pin);

  // initialize primitive variables
  par_for(
      "KHI", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // Calculating the cell center coordinates
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real x2v = CellCenterX(j - js, nx2, x2min, x2max);

        Real dens, pres, vx, vy, vz, scal1, scal2;

        if (iprob == 1) {
          pres = p_in;
          dens = rho0 - rho1 * tanh((x1v - x_cold) / a_char);
          vy = vshear_half +
               vshear_delta * tanh((x1v - x_cold) /
                                   a_char); // this makes relative shear
                                            // velocity = vx_hot - vx_cold.
          // Adding perturbations to vx. The perturbation is a sum of sine
          // functions with different wavelengths. wavenumbers are k_n =
          // 2n*pi/L_y, where n = 5,10,18,25,32.
          Real perturb = sin(2.0 * 5.0 * M_PI * x2v / L_y) +
                         sin(2.0 * 10.0 * M_PI * x2v / L_y) +
                         sin(2.0 * 18.0 * M_PI * x2v / L_y) +
                         sin(2.0 * 25.0 * M_PI * x2v / L_y) +
                         sin(2.0 * 32.0 * M_PI * x2v / L_y);
          vx = -amp * 2.0 * vshear_delta *
               (perturb)*exp(-SQR((x1v - x_cold) / sigma));
          vz = 0.0;
          scal1 = y0 - y1 * tanh((x1v - x_cold) / a_char);
          scal2 = (x1v < x_cold) ? 1.0 : 0.0;
        }

        // setting primitives
        w0_(m, IDN, k, j, i) = dens;
        w0_(m, IEN, k, j, i) = pres / gm1;
        w0_(m, IVX, k, j, i) = vx;
        w0_(m, IVY, k, j, i) = vy;
        w0_(m, IVZ, k, j, i) = vz;
        // adding passive scalars
        w0_(m, nfluid, k, j, i) = scal1;
        w0_(m, nfluid + 1, k, j, i) = scal2;
      });

  // Convert primitives to conserved
  if (pmbp->phydro != nullptr) {
    auto &u0_ = pmbp->phydro->u0;
    pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
  }

  return;
}

void constant_bcs(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  MeshBlockPack *pmbp = pm->pmb_pack;

  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;

  DvceArray5D<Real> u0_, w0_;
  u0_ = pm->pmb_pack->phydro->u0;
  w0_ = pm->pmb_pack->phydro->w0;
  int nmb = pm->pmb_pack->nmb_thispack;
  int &nfluid = pmbp->phydro->nhydro;
  int &nscalars = pmbp->phydro->nscalars;

  // ConsToPrim over all X1 ghost zones *and* at the innermost/outermost
  // X1-active zones of Meshblocks, even if Meshblock face is not at the edge of
  // computational domain
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, is - ng, is,
                                           0, (n2 - 1), 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_, w0_, false, ie, ie + ng,
                                           0, (n2 - 1), 0, (n3 - 1));
  }

  par_for(
      "kh_bcs", DevExeSpace(), 0, (nmb - 1), 0, (n3 - 1), 0, (n2 - 1),
      KOKKOS_LAMBDA(int m, int k, int j) {
        if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
          for (int i = 0; i < ng; ++i) {
            w0_(m, IDN, k, j, is - i - 1) = density_cold;
            w0_(m, IEN, k, j, is - i - 1) = pressure / gm1;
            w0_(m, IVY, k, j, is - i - 1) = velocityy_cold;
            w0_(m, IVX, k, j, is - i - 1) = w0_(m, IVX, k, j, is); // outflow
            w0_(m, IVZ, k, j, is - i - 1) = w0_(m, IVZ, k, j, is);
            w0_(m, nfluid, k, j, is - i - 1) = scalar_cold;
            w0_(m, nfluid + 1, k, j, is - i - 1) = 1.0;
          }
        }
        if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
          for (int i = 0; i < ng; ++i) {
            w0_(m, IDN, k, j, ie + i + 1) = density_hot;
            w0_(m, IEN, k, j, ie + i + 1) = pressure / gm1;
            w0_(m, IVY, k, j, ie + i + 1) = velocityy_hot;
            w0_(m, IVX, k, j, ie + i + 1) = w0_(m, IVX, k, j, ie); // outflow
            w0_(m, IVZ, k, j, ie + i + 1) = w0_(m, IVZ, k, j, ie);
            w0_(m, nfluid, k, j, ie + i + 1) = scalar_hot;
            w0_(m, nfluid + 1, k, j, ie + i + 1) = 0.0;
          }
        }
      });

  // Convert primitives to conserved
  // PrimToCons on X1 ghost zones
  if (pm->pmb_pack->phydro != nullptr) {
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, is - ng, is - 1,
                                           0, (n2 - 1), 0, (n3 - 1));
    pm->pmb_pack->phydro->peos->PrimToCons(w0_, u0_, ie + 1, ie + ng,
                                           0, (n2 - 1), 0, (n3 - 1));
  }
}

// All source terms
namespace {
py::scoped_interpreter *pguard = nullptr;
py::object *psource_func = nullptr;

void UserSourceTerm(Mesh *pm, const Real bdt) {

  if (pguard == nullptr) {
    pguard = new py::scoped_interpreter();
    psource_func = new py::object(py::module_::import("source_module").attr("source_func"));
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &u0 = pmbp->phydro->u0;
  const auto &w0 = pmbp->phydro->w0;
  int nfluid = pmbp->phydro->nhydro;
  int tracer_index = nfluid;
  int frho_index = nfluid + 1;
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;

  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks;
  int nmb = pmbp->nmb_thispack;

  int Ni = ie - is + 1;
  int Nj = je - js + 1;
  int N2D = Ni * Nj * nmb;

  // Device views to collect flattened data
  Kokkos::View<double *> dens_d("dens_d", N2D);
  Kokkos::View<double *> press_d("press_d", N2D);
  Kokkos::View<double *> vx_d("vx_d", N2D);
  Kokkos::View<double *> vy_d("vy_d", N2D);
  Kokkos::View<double *> tracer_d("tracer_d", N2D);
  Kokkos::View<double *> fmclrho_d("fmclrho_d", N2D);

  // Fill the device views using Kokkos
  par_for(
      "FlattenSnapshot", DevExeSpace(), 0, nmb - 1, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int j, int i) {
        int idx = m * (Nj * Ni) + (i - is) * Nj + (j - js);
        dens_d(idx) = w0(m, IDN, ks, j, i);
        press_d(idx) = w0(m, IEN, ks, j, i) * gm1;
        vx_d(idx) = w0(m, IVX, ks, j, i);
        vy_d(idx) = w0(m, IVY, ks, j, i);
        tracer_d(idx) = w0(m, tracer_index, ks, j, i);
        fmclrho_d(idx) = w0(m, frho_index, ks, j, i);
      });

  // Allocate host mirrors and copy to host
  auto dens_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dens_d);
  auto press_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), press_d);
  auto vx_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vx_d);
  auto vy_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vy_d);
  auto tracer_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tracer_d);
  auto fmclrho_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fmclrho_d);

  // Wrap Kokkos::View data in py::array without copy
  py::array_t<double> S_arr =
      (*psource_func)(py::array_t<double>({nmb * Ni, Nj}, dens_h.data()),
                      py::array_t<double>({nmb * Ni, Nj}, press_h.data()),
                      py::array_t<double>({nmb * Ni, Nj}, vx_h.data()),
                      py::array_t<double>({nmb * Ni, Nj}, vy_h.data()),
                      py::array_t<double>({nmb * Ni, Nj}, tracer_h.data()),
                      py::array_t<double>({nmb * Ni, Nj}, fmclrho_h.data()));

  auto S_buf = S_arr.unchecked<2>();

  // Apply source back using Kokkos
  par_for(
      "ApplySrc", DevExeSpace(), 0, nmb - 1, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int j, int i) {
        int idx = m * (Nj * Ni) + (i - is) * Nj + (j - js);

        // rho update
        u0(m, IDN, ks, j, i) =
            fmax(0.0, u0(m, IDN, ks, j, i) + bdt * S_buf(0, idx));

        // momentum update
        u0(m, IM1, ks, j, i) += bdt * S_buf(1, idx);
        u0(m, IM2, ks, j, i) += bdt * S_buf(2, idx);

        // energy update
        u0(m, IEN, ks, j, i) += bdt * S_buf(3, idx);

        // fmcl update
        u0(m, frho_index, ks, j, i) =
            fmax(0.0, u0(m, frho_index, ks, j, i) + bdt * S_buf(4, idx));
      });

  return;
}

void SubgridFinalize(ParameterInput *pin, Mesh *pm) {
  if (psource_func != nullptr) {
    delete psource_func;
    psource_func = nullptr;
  }
  if (pguard != nullptr) {
    delete pguard;
    pguard = nullptr;
  }
}
} // namespace
