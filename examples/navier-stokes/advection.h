// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <math.h>

// *****************************************************************************
// This QFunction sets the the initial conditions and boundary conditions
//
// These initial conditions are given in terms of potential temperature and
//   Exner pressure and potential temperature and then converted to density and
//   total energy. Initial velocity and momentum is zero.
//
// Initial Conditions:
//   Mass:
//     Constant mass of 1.0
//   Momentum:
//     Rotational field in x,y with no momentum in z
//   Energy:
//     Maximum of 1. x0 decreasing linearly to 0. as radial distance increases
//       to 1/8, then 0. everywhere else
//
//  Boundary Conditions:
//    Mass:
//      0.0 flux
//    Momentum:
//      0.0
//    Energy:
//      0.0 flux
//
// *****************************************************************************
static int ICsAdvection(void *ctx, CeedInt Q,
                        const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *coords = in[0];
  // Outputs
  CeedScalar *q0 = out[0], *coordsout = out[1];
  // Setup
  const CeedScalar tol = 1.e-14;
  const CeedScalar x0[3] = {0.25, 0.5, 0.5};
  const CeedScalar center[3] = {0.5, 0.5, 0.5};

  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Coordinates
    const CeedScalar x = coords[i+Q*0];
    const CeedScalar y = coords[i+Q*1];
    const CeedScalar z = coords[i+Q*2];
    // -- Energy
    const CeedScalar r = sqrt(pow((x - x0[0]), 2) +
                              pow((y - x0[1]), 2) +
                              pow((z - x0[2]), 2));

    // Initial Conditions
    q0[i+0*Q] = 1.;
    q0[i+1*Q] = -0.5*(y - center[0]);
    q0[i+2*Q] =  0.5*(x - center[1]);
    q0[i+3*Q] = 0.0;
    q0[i+4*Q] = r <= 1./8. ? (1.-8.*r) : 0.;

    // Homogeneous Dirichlet Boundary Conditions for Momentum
    if ( fabs(x - 0.0) < tol || fabs(x - 1.0) < tol
      || fabs(y - 0.0) < tol || fabs(y - 1.0) < tol
      || fabs(z - 0.0) < tol || fabs(z - 1.0) < tol ) {
      q0[i+1*Q] = 0.0;
      q0[i+2*Q] = 0.0;
      q0[i+3*Q] = 0.0;
    }

    // Coordinates
    coordsout[i+0*Q] = x;
    coordsout[i+1*Q] = y;
    coordsout[i+2*Q] = z;

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the following formulation of the advection equation
//
// This is 3D advection given in two formulations based upon the weak form.
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Density
//   Ui  - Momentum    ,  Ui = rho ui
//   E   - Total Energy,  E  = rho Cv T + rho (u u) / 2 + rho g z
//
// Advection Equation:
//   dE/dt + div( E u ) = 0
//
// *****************************************************************************
static int Advection(void *ctx, CeedInt Q,
                     const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *q = in[0], *dq = in[1], *qdata = in[2], *x = in[3];
  // Outputs
  CeedScalar *v = out[0], *dv = out[1];

  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho     =   q[i+0*Q];
    const CeedScalar u[3]    = { q[i+1*Q] / rho,
                                 q[i+2*Q] / rho,
                                 q[i+3*Q] / rho };
    const CeedScalar E       =   q[i+4*Q];
    // -- Grad in
    const CeedScalar drho[3] = {  dq[i+(0+5*0)*Q],
                                  dq[i+(0+5*1)*Q],
                                  dq[i+(0+5*2)*Q] };
    const CeedScalar du[9]   = { (dq[i+(1+5*0)*Q] - drho[0]*u[0]) / rho,
                                 (dq[i+(1+5*1)*Q] - drho[1]*u[0]) / rho,
                                 (dq[i+(1+5*2)*Q] - drho[2]*u[0]) / rho,
                                 (dq[i+(2+5*0)*Q] - drho[0]*u[1]) / rho,
                                 (dq[i+(2+5*1)*Q] - drho[1]*u[1]) / rho,
                                 (dq[i+(2+5*2)*Q] - drho[2]*u[1]) / rho,
                                 (dq[i+(3+5*0)*Q] - drho[0]*u[2]) / rho,
                                 (dq[i+(3+5*1)*Q] - drho[1]*u[2]) / rho,
                                 (dq[i+(3+5*2)*Q] - drho[2]*u[2]) / rho };
    const CeedScalar dE[3]   = {  dq[i+(4+5*0)*Q],
                                  dq[i+(4+5*1)*Q],
                                  dq[i+(4+5*2)*Q] };
    // -- Interp-to-Interp qdata
    const CeedScalar wJ       =   qdata[i+ 0*Q];
    // -- Interp-to-Grad qdata
    //      Symmetric 3x3 matrix
    const CeedScalar wBJ[9]   = { qdata[i+ 1*Q],
                                  qdata[i+ 2*Q],
                                  qdata[i+ 3*Q],
                                  qdata[i+ 4*Q],
                                  qdata[i+ 5*Q],
                                  qdata[i+ 6*Q],
                                  qdata[i+ 7*Q],
                                  qdata[i+ 8*Q],
                                  qdata[i+ 9*Q] };
    // -- Grad-to-Grad qdata
    const CeedScalar wBBJ[6]  = { qdata[i+10*Q],
                                  qdata[i+11*Q],
                                  qdata[i+12*Q],
                                  qdata[i+13*Q],
                                  qdata[i+14*Q],
                                  qdata[i+15*Q] };

    for (int c=0; c<5; c++) {
      v[c*Q+i] = 0;
      for (int d=0; d<3; d++)
        dv[(d*5+c)*Q+i] = 0;
    }
    // The Physics

    // -- Total Energy
    // ---- Version 1: E u 
    if (1) {
    dv[i+(4+5*0)*Q]  = E*(u[0]*wBJ[0] + u[1]*wBJ[1] + u[2]*wBJ[2]);
    dv[i+(4+5*1)*Q]  = E*(u[0]*wBJ[3] + u[1]*wBJ[4] + u[2]*wBJ[5]);
    dv[i+(4+5*2)*Q]  = E*(u[0]*wBJ[6] + u[1]*wBJ[7] + u[2]*wBJ[8]);
    }
    // ---- Version 2: E du
    if (0) {
    v[i+4*Q]   = E*(du[0]*wBJ[0] + du[3]*wBJ[1] + du[6]*wBJ[2]);
    v[i+4*Q]  -= E*(du[1]*wBJ[3] + du[4]*wBJ[4] + du[7]*wBJ[5]);
    v[i+4*Q]  -= E*(du[2]*wBJ[6] + du[5]*wBJ[7] + du[8]*wBJ[8]);
    }

  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************