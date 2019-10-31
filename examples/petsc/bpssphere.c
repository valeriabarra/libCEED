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

//                        libCEED + PETSc Example: CEED BPs
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP benchmark problems, see http://ceed.exascaleproject.org/bps,
// on a closed surface, such as the one of a sphere.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with:
//
//     make bpsdmplex [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     bpsdmplexsphere -problem bp1 -degree 3
//     bpsdmplexsphere -problem bp2 -ceed /cpu/self -degree 3
//     bpsdmplexsphere -problem bp3 -ceed /cpu/self -degree 3
//     bpsdmplexsphere -problem bp4 -ceed /cpu/occa -degree 3
//     bpsdmplexsphere -problem bp5 -ceed /cpu/occa -degree 3
//     bpsdmplexsphere -problem bp6 -ceed /cpu/self -degree 3
//
//TESTARGS -ceed {ceed_resource} -test -problem bp1 -_degree 3

/// @file
/// CEED BPs example using PETSc with DMPlex
/// See bps.c for a "raw" implementation using a structured grid.
/// and bpsdmplex.c for an implementation using an unstructured grid.
static const char help[] = "Solve CEED BPs on a sphere using DMPlex in PETSc\n";

#include <stdbool.h>
#include <string.h>
#include <petscksp.h>
#include <petscdmplex.h>
#include <ceed.h>
#include "setupsphere.h"

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char ceedresource[PETSC_MAX_PATH_LEN] = "/cpu/self",
                                          filename[PETSC_MAX_PATH_LEN];
//  double my_rt_start, my_rt, rt_min, rt_max;
  PetscInt degree, qextra, lsize, gsize, ncompx = 3, ncompu = 1,
//           cStart, cEnd, nelem,
           xlsize, topodim;
//  PetscScalar *r;
//  const PetscScalar *coordArray;
  PetscBool test_mode, benchmark_mode, read_mesh,
            write_solution, simplex;
  Vec X, Xloc, V, Vloc;
//      , coords;
  Mat matO;
  KSP ksp;
  DM  dm;
//      , dmcoord;
//  PetscSpace sp;
//  PetscFE fe;
//  PetscSection section;
  UserO userO;
  Ceed ceed;
  CeedData ceeddata;
//  CeedBasis basisx, basisu;
//  CeedElemRestriction Erestrictx, Erestrictu, Erestrictxi, Erestrictui,
//                      Erestrictqdi;
//  CeedQFunction qf_error; // qf_setup, qf_apply,
//  CeedOperator op_error; // op_setup, op_apply,
//  CeedVector rhsceed, target; // xcoord, qdata,
  bpType bpChoice;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  // Read CL options
  ierr = PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL); CHKERRQ(ierr);
  bpChoice = CEED_BP1;
  ierr = PetscOptionsEnum("-problem",
                          "CEED benchmark problem to solve", NULL,
                          bpTypes, (PetscEnum)bpChoice, (PetscEnum *)&bpChoice,
                          NULL); CHKERRQ(ierr);
  ncompu = bpOptions[bpChoice].ncompu;
  test_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, test_mode, &test_mode, NULL); CHKERRQ(ierr);
  benchmark_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-benchmark",
                          "Benchmarking mode (prints benchmark statistics)",
                          NULL, benchmark_mode, &benchmark_mode, NULL);
  CHKERRQ(ierr);
  write_solution = PETSC_FALSE;
  ierr = PetscOptionsBool("-write_solution",
                          "Write solution for visualization",
                          NULL, write_solution, &write_solution, NULL);
  CHKERRQ(ierr);
  degree = test_mode ? 3 : 2;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  qextra = bpOptions[bpChoice].qextra;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  read_mesh = PETSC_FALSE;
  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            filename, filename, sizeof(filename), &read_mesh);
  CHKERRQ(ierr);
  topodim = 2;
  ierr = PetscOptionsInt("-topodim", "Topological dimension",
                         NULL, topodim, &topodim, NULL); CHKERRQ(ierr);
  if (topodim != 2)
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "Unsupported dimension for sphere: %D. It must be 2.\n", topodim);
  simplex = PETSC_FALSE;
  ierr = PetscOptionsBool("-simplex", "Use simplices, or tensor product cells",
                         NULL, simplex, &simplex, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Setup DM
  if (read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, PETSC_TRUE, &dm);
    CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateSphereMesh(PETSC_COMM_WORLD, topodim, simplex, &dm); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) dm, "Sphere"); CHKERRQ(ierr);
    // Distribute mesh over processes
    {
      DM dmDist = NULL;
      PetscPartitioner part;

      ierr = DMPlexGetPartitioner(dm, &part); CHKERRQ(ierr);
      ierr = PetscPartitionerSetFromOptions(part); CHKERRQ(ierr);
      ierr = DMPlexDistribute(dm, 0, NULL, &dmDist); CHKERRQ(ierr);
      if (dmDist) {
        // Debug
//        DMView(dmDist,PETSC_VIEWER_STDOUT_WORLD);
        // End Debug
        ierr = DMDestroy(&dm); CHKERRQ(ierr);
        dm  = dmDist;
      }
    }
    // Refine DMPlex with uniform refinement
    ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE);
    ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);
    ierr = ProjectToUnitSphere(dm); CHKERRQ(ierr);
  }
//  ierr = PetscFECreateDefault(PETSC_COMM_SELF, topodim, ncompu, PETSC_FALSE, NULL,
//                              PETSC_DETERMINE, &fe);
//  CHKERRQ(ierr);
//  ierr = DMAddField(dm, NULL, (PetscObject)fe); CHKERRQ(ierr);
//  ierr = DMCreateDS(dm); CHKERRQ(ierr);
//  ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL);
//  CHKERRQ(ierr);

//  // Get basis space degree
//  ierr = PetscFEGetBasisSpace(fe, &sp); CHKERRQ(ierr);
//  ierr = PetscSpaceGetDegree(sp, &degree, NULL); CHKERRQ(ierr);
//  ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);
//  if (degree < 1) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
//                             "-degree %D must be at least 1", degree);

  // Create DM
  ierr = SetupDMByDegree(dm, degree, ncompu, topodim, bpChoice);
  CHKERRQ(ierr);

  // Create vectors
  ierr = DMCreateGlobalVector(dm, &X); CHKERRQ(ierr);
  ierr = VecGetLocalSize(X, &lsize); CHKERRQ(ierr);
  ierr = VecGetSize(X, &gsize); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &Xloc); CHKERRQ(ierr);
  ierr = VecGetSize(Xloc, &xlsize); CHKERRQ(ierr);
  ierr = VecDuplicate(X, &V); CHKERRQ(ierr);
  ierr = VecDuplicate(Xloc, &Vloc); CHKERRQ(ierr);

  // Operator
  ierr = PetscMalloc1(1, &userO); CHKERRQ(ierr);
  ierr = MatCreateShell(comm, lsize, lsize, gsize, gsize,
                        userO, &matO); CHKERRQ(ierr);
  ierr = MatShellSetOperation(matO, MATOP_MULT,
                              (void(*)(void))MatMult_Ceed);
  CHKERRQ(ierr);

  // Print summary
  if (!test_mode) {
    PetscInt P = degree + 1, Q = P + qextra;
    ierr = PetscPrintf(comm,
                       "\n-- CEED Benchmark Problem %d -- libCEED + PETSc --\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (p)       : %d\n"
                       "    Number of 1D Quadrature Points (q) : %d\n"
                       "    Global nodes                       : %D\n",
                       bpChoice+1, ceedresource, P, Q,  gsize/ncompu);
    CHKERRQ(ierr);
  }

  // Set up libCEED
  CeedInit(ceedresource, &ceed);

  // Setup libCEED's objects
  ierr = PetscMalloc1(1, &ceeddata); CHKERRQ(ierr);
  ierr = SetupLibceedByDegree(dm, ceed, degree, topodim, qextra,
                              ncompx, ncompu, gsize, xlsize, bpChoice,
                              ceeddata); CHKERRQ(ierr);

//  // Create the error Q-function
//  CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpChoice].error,
//                              bpOptions[bpChoice].errorfname, &qf_error);
//  CeedQFunctionAddInput(qf_error, "u", ncompu, CEED_EVAL_INTERP);
//  CeedQFunctionAddInput(qf_error, "true_soln", ncompu, CEED_EVAL_NONE);
//  CeedQFunctionAddOutput(qf_error, "error", ncompu, CEED_EVAL_NONE);

//  // Create the error operator
//  CeedOperatorCreate(ceed, qf_error, NULL, NULL, &op_error);
//  CeedOperatorSetField(op_error, "u", ceeddata->Erestrictu,
//                       CEED_TRANSPOSE, ceeddata->basisu,
//                       CEED_VECTOR_ACTIVE);
//  CeedOperatorSetField(op_error, "true_soln", ceeddata->Erestrictui,
//                       CEED_NOTRANSPOSE, CEED_BASIS_COLLOCATED, target);
//  CeedOperatorSetField(op_error, "error", ceeddata->Erestrictui,
//                       CEED_NOTRANSPOSE, CEED_BASIS_COLLOCATED,
//                       CEED_VECTOR_ACTIVE);

  // Set up Mat
  userO->comm = comm;
  userO->dm = dm;
  userO->Xloc = Xloc;
  ierr = VecDuplicate(Xloc, &userO->Yloc); CHKERRQ(ierr);
  userO->xceed = ceeddata->xceed;
  userO->yceed = ceeddata->yceed;
  userO->op = ceeddata->op_apply;
  userO->ceed = ceed;
  userO->topodim = topodim;
  userO->simplex = simplex;

  // Setup solver
//  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
//  {
//    PC pc;
//    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
//    if (bpChoice == CEED_BP1 || bpChoice == CEED_BP2) {
//      ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
//      ierr = PCJacobiSetType(pc, PC_JACOBI_ROWSUM); CHKERRQ(ierr);
//    } else {
//      ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
//    }
//    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
//    ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL); CHKERRQ(ierr);
//    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
//                            PETSC_DEFAULT); CHKERRQ(ierr);
//  }
//  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
//  ierr = KSPSetOperators(ksp, matO, matO); CHKERRQ(ierr);

//  // First run, if benchmarking
//  if (benchmark_mode) {
//    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1);
//    CHKERRQ(ierr);
//    my_rt_start = MPI_Wtime();
//    ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
//    my_rt = MPI_Wtime() - my_rt_start;
//    ierr = MPI_Allreduce(MPI_IN_PLACE, &my_rt, 1, MPI_DOUBLE, MPI_MIN, comm);
//    CHKERRQ(ierr);
//    // Set maxits based on first iteration timing
//    if (my_rt > 0.02) {
//      ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 5);
//      CHKERRQ(ierr);
//    } else {
//      ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 20);
//      CHKERRQ(ierr);
//    }
//  }

//  // Timed solve
//  ierr = VecZeroEntries(X); CHKERRQ(ierr);
//  ierr = PetscBarrier((PetscObject)ksp); CHKERRQ(ierr);
//  my_rt_start = MPI_Wtime();
//  ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
//  my_rt = MPI_Wtime() - my_rt_start;

//  // Output results
//  {
//    KSPType ksptype;
//    KSPConvergedReason reason;
//    PetscReal rnorm;
//    PetscInt its;
//    ierr = KSPGetType(ksp, &ksptype); CHKERRQ(ierr);
//    ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
//    ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
//    ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
//    if (!test_mode || reason < 0 || rnorm > 1e-8) {
//      ierr = PetscPrintf(comm,
//                         "  KSP:\n"
//                         "    KSP Type                           : %s\n"
//                         "    KSP Convergence                    : %s\n"
//                         "    Total KSP Iterations               : %D\n"
//                         "    Final rnorm                        : %e\n",
//                         ksptype, KSPConvergedReasons[reason], its,
//                         (double)rnorm); CHKERRQ(ierr);
//    }
//    if (!test_mode) {
//      ierr = PetscPrintf(comm,"  Performance:\n"); CHKERRQ(ierr);
//    }
//    {
//      PetscReal maxerror;
//      ierr = ComputeErrorMax(userO, op_error, X, target, &maxerror);
//      CHKERRQ(ierr);
//      PetscReal tol = 5e-2;
//      if (!test_mode || maxerror > tol) {
//        ierr = MPI_Allreduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, comm);
//        CHKERRQ(ierr);
//        ierr = MPI_Allreduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, comm);
//        CHKERRQ(ierr);
//        ierr = PetscPrintf(comm,
//                           "    Pointwise Error (max)              : %e\n"
//                           "    CG Solve Time                      : %g (%g) sec\n",
//                           (double)maxerror, rt_max, rt_min); CHKERRQ(ierr);
//      }
//    }
//    if (benchmark_mode && (!test_mode)) {
//      ierr = PetscPrintf(comm,
//                         "    DoFs/Sec in CG                     : %g (%g) million\n",
//                         1e-6*gsize*its/rt_max,
//                         1e-6*gsize*its/rt_min); CHKERRQ(ierr);
//    }
//  }

//  // Output solution
//  if (write_solution) {
//    PetscViewer vtkviewersoln;

//    ierr = PetscViewerCreate(comm, &vtkviewersoln); CHKERRQ(ierr);
//    ierr = PetscViewerSetType(vtkviewersoln, PETSCVIEWERVTK); CHKERRQ(ierr);
//    ierr = PetscViewerFileSetName(vtkviewersoln, "solution.vtu"); CHKERRQ(ierr);
//    ierr = VecView(X, vtkviewersoln); CHKERRQ(ierr);
//    ierr = PetscViewerDestroy(&vtkviewersoln); CHKERRQ(ierr);
//  }

  // Create auxiliary solution-size vectors
  CeedVector uceed, vceed;
  CeedVectorCreate(ceed, xlsize, &uceed);
  CeedVectorCreate(ceed, xlsize, &vceed);
  PetscScalar *v;
  ierr = VecZeroEntries(Vloc); CHKERRQ(ierr);
  ierr = VecGetArray(Vloc, &v);
  CeedVectorSetArray(vceed, CEED_MEM_HOST, CEED_USE_POINTER, v);

  // Initialize u and v with ones
  CeedVectorSetValue(uceed, 1.0);
  CeedVectorSetValue(vceed, 1.0);

  // Apply the mass operator: 'u' -> 'v'
  CeedOperatorApply(ceeddata->op_apply, uceed, vceed, CEED_REQUEST_IMMEDIATE);

  // Gather output vector
  ierr = VecRestoreArray(Vloc, &v); CHKERRQ(ierr);
  ierr = VecZeroEntries(V); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, Vloc, ADD_VALUES, V); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, Vloc, ADD_VALUES, V); CHKERRQ(ierr);

  // Compute and print the sum of the entries of 'v' giving the mesh surface area
  PetscScalar area;
  ierr = VecSum(V, &area); CHKERRQ(ierr);

  CeedScalar R = 1,                      // radius of the sphere
             l = 1.0/PetscSqrtReal(3.0); // edge of the inscribed cube
  CeedScalar exact_surfarea = 6 * (2*l) * (2*l); // for refine 0, i.e., a cube
#ifndef M_PI
  #define M_PI    3.14159265358979323846
#endif
//    CeedScalar exact_surfarea = 4 * M_PI; // sphere has radius R=1

  ierr = PetscPrintf(comm, " done.\n"); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Exact mesh surface area    : % .14g\n",
                     exact_surfarea); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Computed mesh surface area : % .14g\n", area);
  CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Area error                 : % .14g\n",
                     fabs(area - exact_surfarea)); CHKERRQ(ierr);

  // Cleanup
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = VecDestroy(&Xloc); CHKERRQ(ierr);
  ierr = VecDestroy(&userO->Yloc); CHKERRQ(ierr);
  ierr = MatDestroy(&matO); CHKERRQ(ierr);
  ierr = PetscFree(userO); CHKERRQ(ierr);
  ierr = CeedDataDestroy(0, ceeddata); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = VecDestroy(&V); CHKERRQ(ierr);
  ierr = VecDestroy(&Vloc); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  CeedVectorDestroy(&uceed);
  CeedVectorDestroy(&vceed);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
