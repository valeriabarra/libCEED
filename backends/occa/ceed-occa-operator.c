// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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
#define CEED_DEBUG_COLOR 198
#include "ceed-occa.h"

// *****************************************************************************
// * Destroy the CeedOperator_Occa
// *****************************************************************************
static int CeedOperatorDestroy_Occa(CeedOperator op) {
  CeedOperator_Occa *impl = op->data;
  int ierr;

  for (CeedInt i=0; i<impl->numein+impl->numeout; i++) {
    ierr = CeedVectorDestroy(&impl->evecs[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->evecs); CeedChk(ierr);
  ierr = CeedFree(&impl->edata); CeedChk(ierr);

  for (CeedInt i=0; i<impl->numqin+impl->numqout; i++) {
    ierr = CeedFree(&impl->qdata_alloc[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&impl->qdata_alloc); CeedChk(ierr);
  ierr = CeedFree(&impl->qdata); CeedChk(ierr);

  ierr = CeedFree(&impl->indata); CeedChk(ierr);
  ierr = CeedFree(&impl->outdata); CeedChk(ierr);

  ierr = CeedFree(&op->data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Setup infields or outfields
// *****************************************************************************
static int CeedOperatorSetupFields_Occa(struct CeedQFunctionField qfields[16],
                                       struct CeedOperatorField ofields[16],
                                       CeedVector *evecs, CeedScalar **qdata,
                                       CeedScalar **qdata_alloc, CeedScalar **indata,
                                       CeedInt starti, CeedInt starte,
                                       CeedInt startq, CeedInt numfields,
                                       CeedInt Q, CeedInt nelem) {
  CeedInt dim, ierr, ie=starte, iq=startq, ncomp;

  // Loop over fields
  for (CeedInt i=0; i<numfields; i++) {
    if (ofields[i].Erestrict) {
      ierr = CeedElemRestrictionCreateVector(ofields[i].Erestrict, NULL, &evecs[ie]);
      CeedChk(ierr);
      ie++;
    }
    CeedEvalMode emode = qfields[i].emode;
    switch(emode) {
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_INTERP:
      ncomp = qfields[i].ncomp;
      nelem = 
      ierr = CeedMalloc(Q*ncomp*nelem, &qdata_alloc[iq]); CeedChk(ierr);
      qdata[i + starti] = qdata_alloc[iq];
      iq++;
      break;
    case CEED_EVAL_GRAD:
      ncomp = qfields[i].ncomp;
      dim = ofields[i].basis->dim;
      ierr = CeedMalloc(Q*ncomp*dim*nelem, &qdata_alloc[iq]); CeedChk(ierr);
      qdata[i + starti] = qdata_alloc[iq];
      iq++;
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedMalloc(Q, &qdata_alloc[iq]); CeedChk(ierr);
      ierr = CeedBasisApply(ofields[iq].basis, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                            NULL, qdata_alloc[iq]); CeedChk(ierr);
      qdata[i] = qdata_alloc[iq];
      indata[i] = qdata[i];
      iq++;
      break;
    case CEED_EVAL_DIV:
      break; // Not implimented
    case CEED_EVAL_CURL:
      break; // Not implimented
    }
  }
  return 0;
}

// *****************************************************************************
// * CeedOperator needs to connect all the named fields (be they active or passive)
// * to the named inputs and outputs of its CeedQFunction.
// *****************************************************************************
static int CeedOperatorSetup_Occa(CeedOperator op) {
  if (op->setupdone) return 0;
  CeedOperator_Occa *opocca = op->data;
  CeedQFunction qf = op->qf;
  CeedInt Q = op->numqpoints;
  int ierr;

  // Count infield and outfield array sizes and evectors
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    CeedEvalMode emode = qf->inputfields[i].emode;
    opocca->numqin += !!(emode & CEED_EVAL_INTERP) + !!(emode & CEED_EVAL_GRAD) + !!
                     (emode & CEED_EVAL_WEIGHT);
    opocca->numein +=
      !!op->inputfields[i].Erestrict; // Need E-vector when restriction exists
  }
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    CeedEvalMode emode = qf->outputfields[i].emode;
    opocca->numqout += !!(emode & CEED_EVAL_INTERP) + !!(emode & CEED_EVAL_GRAD);
    opocca->numeout += !!op->outputfields[i].Erestrict;
  }

  // Allocate
  ierr = CeedCalloc(opocca->numein + opocca->numeout, &opocca->evecs); CeedChk(ierr);
  ierr = CeedCalloc(qf->numinputfields + qf->numoutputfields, &opocca->edata);
  CeedChk(ierr);

  ierr = CeedCalloc(opocca->numqin + opocca->numqout, &opocca->qdata_alloc);
  CeedChk(ierr);
  ierr = CeedCalloc(qf->numinputfields + qf->numoutputfields, &opocca->qdata);
  CeedChk(ierr);

  ierr = CeedCalloc(16, &opocca->indata); CeedChk(ierr);
  ierr = CeedCalloc(16, &opocca->outdata); CeedChk(ierr);

  // Set up infield and outfield pointer arrays
  // Infields
  ierr = CeedOperatorSetupFields_Occa(qf->inputfields, op->inputfields,
                                     opocca->evecs, opocca->qdata, opocca->qdata_alloc,
                                     opocca->indata, 0, 0, 0,
                                     qf->numinputfields, Q,
                                     op->numelements); CeedChk(ierr);

  // Outfields
  ierr = CeedOperatorSetupFields_Occa(qf->outputfields, op->outputfields,
                                     opocca->evecs, opocca->qdata, opocca->qdata_alloc,
                                     opocca->indata, qf->numinputfields, opocca->numein,
                                     opocca->numqin, qf->numoutputfields, Q,
                                     op->numelements); CeedChk(ierr);

  op->setupdone = 1;

  return 0;
}

// *****************************************************************************
// * Apply CeedOperator to a vector
// *****************************************************************************
static int CeedOperatorApply_Occa(CeedOperator op,
                                  CeedVector ustate,
                                  CeedVector residual, CeedRequest *request) {
  const Ceed ceed = op->ceed;
  dbg("[CeedOperator][Apply]");
  CeedOperator_Occa *opocca = op->data;
  CeedBasis basis;
  CeedEvalMode emode;
  CeedInt Q = op->numqpoints, elemsize;
  int ierr;
  CeedQFunction qf = op->qf;
  CeedTransposeMode lmode = CEED_NOTRANSPOSE;
  char *qd;

  // Setup *********************************************************************
  ierr = CeedOperatorSetup_Occa(op); CeedChk(ierr);

  // Input Evecs, Restriction, and Basis action
  for (CeedInt i=0,iein=0; i<qf->numinputfields; i++) {
    // Restriction
    if (op->inputfields[i].Erestrict) {
      // Passive
      if (op->inputfields[i].vec) {
        ierr = CeedElemRestrictionApply(op->inputfields[i].Erestrict, CEED_NOTRANSPOSE,
                                        lmode, op->inputfields[i].vec, opocca->evecs[iein],
                                        request); CeedChk(ierr);
        ierr = CeedVectorGetArrayRead(opocca->evecs[iein], CEED_MEM_HOST,
                                      (const CeedScalar **) &opocca->edata[i]); CeedChk(ierr);
        iein++;
      } else {
        // Active
        ierr = CeedElemRestrictionApply(op->inputfields[i].Erestrict, CEED_NOTRANSPOSE,
                                        lmode, invec, opocca->evecs[iein], request); CeedChk(ierr);
        ierr = CeedVectorGetArrayRead(opocca->evecs[iein], CEED_MEM_HOST,
                                      (const CeedScalar **) &opocca->edata[i]); CeedChk(ierr);
        iein++;
      }
    } else {
      // No restriction
      emode = qf->inputfields[i].emode;
      if (emode & CEED_EVAL_WEIGHT) {
      } else {
        ierr = CeedVectorGetArrayRead(op->inputfields[i].vec, CEED_MEM_HOST,
                                      (const CeedScalar **) &opocca->edata[i]); CeedChk(ierr);
      }
    }
    // Apply input Basis action
    basis = op->inputfields[i].basis;
    emode = qf->outputfields[i].emode;
    if (basis) {
      // Basis action
      switch(emode) {
      case CEED_EVAL_NONE:
        opocca->indata[i] = opocca->edata[i];
        break;
      case CEED_EVAL_INTERP:
        ierr = CeedBasisApplyElems_Occa(op->inputfields[i].basis, Q, CEED_NOTRANSPOSE,
                              CEED_EVAL_INTERP, &opocca->edata[i], opocca->qdata[i]);
        CeedChk(ierr);
        opocca->indata[i] = opocca->qdata[i];
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedBasisApplyElems_Occa(op->inputfields[i].basis, Q, CEED_NOTRANSPOSE,
                              CEED_EVAL_GRAD, &opocca->edata[i], opocca->qdata[i]);
        CeedChk(ierr);
        opocca->indata[i] = opocca->qdata[i];
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
      case CEED_EVAL_DIV:
        break; // Not implimented
      case CEED_EVAL_CURL:
        break; // Not implimented
      }
    }
  }

  // Output Evecs
  for (CeedInt i=0,ieout=opocca->numein; i<qf->numoutputfields; i++) {
    // Restriction
    if (op->outputfields[i].Erestrict) {
      ierr = CeedVectorGetArray(opocca->evecs[ieout], CEED_MEM_HOST,
                                &opocca->edata[i + qf->numinputfields]); CeedChk(ierr);
      ieout++;
    } else {
      // No restriction
      // Passive
      if (op->inputfields[i].vec) {
        ierr = CeedVectorGetArray(op->inputfields[i].vec, CEED_MEM_HOST,
                                  &opocca->edata[i + qf->numinputfields]); CeedChk(ierr);
      } else {
        // Active
        ierr = CeedVectorGetArray(outvec, CEED_MEM_HOST,
                                  &opocca->edata[i + qf->numinputfields]); CeedChk(ierr);
      }
    }
  }

  // Output Qvecs
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    emode = qf->outputfields[i].emode;
    if (emode != CEED_EVAL_NONE) {
      opocca->outdata[i] =  opocca->qdata[i + qf->numinputfields];
    }
  }
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Need to Update this part !!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Before the loop, set up all in/out data on GPU
  // Loop through elements
  for (CeedInt e=0; e<op->numelements; e++) {
    // Output pointers
    for (CeedInt i=0; i<qf->numoutputfields; i++) {
      CeedEvalMode emode = qf->outputfields[i].emode;
      if (emode == CEED_EVAL_NONE) {
        CeedInt ncomp = qf->outputfields[i].ncomp;
        opocca->outdata[i] = &opocca->edata[i + qf->numinputfields][e*Q*ncomp];
      }
    }
    // Q function
    ierr = CeedQFunctionApply(op->qf, Q, (const CeedScalar * const*) opocca->indata,
                              opocca->outdata); CeedChk(ierr);
    }
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// After noop need to get out data from GPU

  // Output Restriction and Basis action
  for (CeedInt i=0,ieout=opocca->numein; i<qf->numoutputfields; i++) {
    // Output basis apply if needed
    // Get emode
    CeedEvalMode emode = qf->outputfields[i].emode;
    // Basis action
    switch(emode) {
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_INTERP:
      ierr = CeedBasisApplyElems_Occa(op->outputfields[i].basis, Q, CEED_TRANSPOSE,
                            CEED_EVAL_INTERP, opocca->outdata[i],
                            &opocca->edata[i + qf->numinputfields]); CeedChk(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedBasisApplyElems_Occa(op->outputfields[i].basis, Q, CEED_TRANSPOSE,
                                      CEED_EVAL_GRAD, opocca->outdata[i],
                                      &opocca->edata[i + qf->numinputfields]);
      CeedChk(ierr);
      break;
    case CEED_EVAL_WEIGHT:
      break; // Should not occur
    case CEED_EVAL_DIV:
      break; // Not implimented
    case CEED_EVAL_CURL:
      break; // Not implimented
    }
    // Restriction
    if (op->outputfields[i].Erestrict) {
      // Passive
      if (op->outputfields[i].vec) {
        ierr = CeedVectorRestoreArray(opocca->evecs[ieout],
                                      &opocca->edata[i + qf->numinputfields]); CeedChk(ierr);
        ierr = CeedElemRestrictionApply(op->outputfields[i].Erestrict, CEED_TRANSPOSE,
                                        lmode, opocca->evecs[ieout], op->outputfields[i].vec, request); CeedChk(ierr);
        ieout++;
      } else {
        // Active
        ierr = CeedVectorRestoreArray(opocca->evecs[ieout],
                                      &opocca->edata[i + qf->numinputfields]); CeedChk(ierr);
        ierr = CeedElemRestrictionApply(op->outputfields[i].Erestrict, CEED_TRANSPOSE,
                                        lmode, opocca->evecs[ieout], outvec, request); CeedChk(ierr);
        ieout++;
      }
    } else {
      // No Restriction
      // Passive
      if (op->outputfields[i].vec) {
        ierr = CeedVectorRestoreArray(op->outputfields[i].vec,
                                      &opocca->edata[i + qf->numinputfields]); CeedChk(ierr);
      } else {
        // Active
        ierr = CeedVectorRestoreArray(outvec, &opocca->edata[i + qf->numinputfields]);
        CeedChk(ierr);
      }
    }
  }

  return 0;
}
/*
  // Fill CeedBasis_Occa's structure with CeedElemRestriction ******************
  CeedBasis_Occa *basis = op->basis->data;
  basis->er = op->Erestrict;
  // Fill CeedQFunction_Occa's structure with nc, dim & qdata ******************
  CeedQFunction_Occa *qfd = op->qf->data;
  qfd->op = true;
  qfd->nc = nc;
  qfd->dim = dim;
  qfd->nelem = nelem;
  qfd->elemsize = elemsize;
  qfd->d_q = ((CeedVector_Occa *)qdata->data)->d_array;
  // ***************************************************************************
  if (!data->etmp) {
    const int n = nc*nelem*elemsize;
    const int bn = Q*nc*(dim+2)*nelem;
    dbg("[CeedOperator][Apply] Setup, n=%d & bn=%d",n,bn);
    ierr = CeedVectorCreate(op->ceed,n,&data->etmp); CeedChk(ierr);
    ierr = CeedVectorCreate(op->ceed,bn,&data->BEu); CeedChk(ierr);
    ierr = CeedVectorCreate(op->ceed,bn,&data->BEv); CeedChk(ierr);
    // etmp is allocated when CeedVectorGetArray is called below
  }
  // Push the memory to the QFunction that will be used
  qfd->b_u = ((CeedVector_Occa *)data->BEu->data)->d_array;
  qfd->b_v = ((CeedVector_Occa *)data->BEv->data)->d_array;
  etmp = data->etmp;
  if (op->qf->inmode & ~CEED_EVAL_WEIGHT) {
    dbg("[CeedOperator][Apply] Apply Restriction");
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_NOTRANSPOSE,
                                    nc, lmode, ustate, etmp,
                                    CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  }
  // We want to avoid Get/Restore
  ierr = CeedVectorGetArray(etmp, CEED_MEM_HOST, &Eu); CeedChk(ierr);
  // Fetching back data from device memory
  ierr = CeedVectorGetArray(qdata, CEED_MEM_HOST, (CeedScalar**)&qd);
  CeedChk(ierr);
  // Local arrays, sizes & pointers ********************************************
  CeedScalar BEu[Q*nc*(dim+2)], BEv[Q*nc*(dim+2)], *out[5] = {0,0,0,0,0};
  const CeedScalar *in[5] = {0,0,0,0,0};
  const size_t qbytes = op->qf->qdatasize;
  // ***************************************************************************
  ierr = CeedBasisApplyElems_Occa(op->basis,Q,CEED_NOTRANSPOSE,op->qf->inmode,
                                  data->etmp,data->BEu); CeedChk(ierr);
  // ***************************************************************************
  dbg("[CeedOperator][Apply] Q for-loop");
  for (CeedInt e=0; e<nelem; e++) {
    for(CeedInt k=0; k<(Q*nc*(dim+2)); k++) BEu[k]=0.0;
    ierr = CeedBasisApply(op->basis, CEED_NOTRANSPOSE,op->qf->inmode,
                          &Eu[e*nc*elemsize], BEu); CeedChk(ierr);
    CeedScalar *u_ptr = BEu, *v_ptr = BEv;
    if (op->qf->inmode & CEED_EVAL_INTERP) { in[0] = u_ptr; u_ptr += Q*nc; }
    if (op->qf->inmode & CEED_EVAL_GRAD) { in[1] = u_ptr; u_ptr += Q*nc*dim; }
    if (op->qf->inmode & CEED_EVAL_WEIGHT) { in[4] = u_ptr; u_ptr += Q; }
    if (op->qf->outmode & CEED_EVAL_INTERP) { out[0] = v_ptr; v_ptr += Q*nc; }
    if (op->qf->outmode & CEED_EVAL_GRAD) { out[1] = v_ptr; v_ptr += Q*nc*dim; }
    qfd->e = e;
    ierr = CeedQFunctionApply(op->qf, &qd[e*Q*qbytes], Q, in, out); CeedChk(ierr);
    ierr = CeedBasisApply(op->basis, CEED_TRANSPOSE,op->qf->outmode, BEv,
                          &Eu[e*nc*elemsize]); CeedChk(ierr);
  }
  // ***************************************************************************
  ierr = CeedBasisApplyElems_Occa(op->basis,Q,CEED_TRANSPOSE,op->qf->outmode,
                                  data->BEv,data->etmp); CeedChk(ierr);
  // *************************************************************************
  ierr = CeedVectorRestoreArray(etmp, &Eu); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(qdata, (CeedScalar**)&qd); CeedChk(ierr);
  // ***************************************************************************
  if (residual) {
    dbg("[CeedOperator][Apply] residual");
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_TRANSPOSE,
                                    nc, lmode, etmp, residual,
                                    CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
    // Restore used pointer if one was provided ********************************
    const CeedVector_Occa *data = residual->data;
    if (data->used_pointer)
      occaCopyMemToPtr(data->used_pointer,data->d_array,
                       residual->length*sizeof(CeedScalar),
                       NO_OFFSET, NO_PROPS);
  }
  // ***************************************************************************
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  return 0;
}
*/

// *****************************************************************************
// * Create an operator
// *****************************************************************************
int CeedOperatorCreate_Occa(CeedOperator op) {
  CeedOperator_Occa *impl;
  int ierr;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  op->data = impl;
  op->Destroy = CeedOperatorDestroy_Occa;
  op->Apply = CeedOperatorApply_Occa;
  return 0;
}
