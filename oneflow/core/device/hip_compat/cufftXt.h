/*
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
*/
#pragma once
#include "oneflow/core/device/cuda_to_hip.h"
#include <cublasLt.h>
#include <hipfft/hipfftXt.h>

inline hipfftResult cufftXtMakePlanMany(hipfftHandle plan, int rank, long long int* n,
                                        long long int* inembed, long long int istride,
                                        long long int idist, hipDataType inputtype,
                                        long long int* onembed, long long int ostride,
                                        long long int odist, hipDataType outputtype,
                                        long long int batch, size_t* work_size,
                                        hipDataType executiontype) {
  // Making a plan also runs work on the null stream (see HandleCreationGuard).
  oneflow_hip_cublaslt::HandleCreationGuard guard;
  return hipfftXtMakePlanMany(plan, rank, n, inembed, istride, idist, inputtype, onembed, ostride,
                              odist, outputtype, batch, work_size, executiontype);
}
