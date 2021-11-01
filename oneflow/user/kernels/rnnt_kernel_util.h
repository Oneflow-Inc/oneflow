/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_USER_KERNELS_RNNT_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_RNNT_KERNEL_UTIL_H_

#include <tuple>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

typedef struct CUstream_st* CUstream;

typedef enum {
    RNNT_STATUS_SUCCESS = 0,
    RNNT_STATUS_MEMOPS_FAILED = 1,
    RNNT_STATUS_INVALID_VALUE = 2,
    RNNT_STATUS_EXECUTION_FAILED = 3,
    RNNT_STATUS_UNKNOWN_ERROR = 4
} rnntStatus_t;

int get_warprnnt_version();

const char* rnntGetStatusString(rnntStatus_t status);

struct rnntOptions {
    unsigned int num_threads;
    int blank_label;
    int maxT;
    int maxU;
    bool batch_first;
};

}

} // oneflow

#endif  // ONEFLOW_USER_KERNELS_RNNT_KERNEL_UTIL_H_