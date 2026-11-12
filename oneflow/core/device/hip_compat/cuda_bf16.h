/*
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
*/
#pragma once
#include "oneflow/core/device/cuda_to_hip.h"
#include <hip/hip_bf16.h>
// CUDA's cuda_bf16.h defines this, and the bf16 kernels and the NCCL bf16 type are gated on
// it; hip_bf16.h has the types but not the macro.
#ifndef __CUDA_BF16_TYPES_EXIST__
#define __CUDA_BF16_TYPES_EXIST__
#endif
