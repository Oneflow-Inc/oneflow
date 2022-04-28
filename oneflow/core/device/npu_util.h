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
#ifdef WITH_NPU

#ifndef ONEFLOW_CORE_DEVICE_NPU_UTIL_H_
#define ONEFLOW_CORE_DEVICE_NPU_UTIL_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/npu/npu_stream.h"

#include "acl/acl.h"
#include "acl/acl_base.h"
#include <iostream>
namespace oneflow {

#define OF_NPU_CHECK(condition)                                                               \
  for (aclError _of_npu_check_status = (condition); _of_npu_check_status != ACL_SUCCESS;) \
  LOG(FATAL) << "Check failed: " #condition " : " << " (" << _of_npu_check_status << ") "

const int32_t kNpuThreadsNumPerBlock = 512;
const int32_t kNpuMaxBlocksNum = 8192;
const int32_t kNpuWarpSize = 32;

// 48KB, max byte size of shared memroy per thread block
// TODO: limit of shared memory should be different for different arch
const int32_t kNpuMaxSharedMemoryByteSize = 48 << 10;


class NpuCurrentDeviceGuard final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NpuCurrentDeviceGuard);
  explicit NpuCurrentDeviceGuard(int32_t dev_id);
  NpuCurrentDeviceGuard();
  ~NpuCurrentDeviceGuard();

 private:
  int32_t saved_dev_id_ = -1;
};

int GetNpuSmVersion();

int GetNpuPtxVersion();

int GetNpuDeviceIndex();

int GetNpuDeviceCount();

void InitNpuContextOnce(int device_id);

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_CORE_DEVICE_CUDA_UTIL_H_
