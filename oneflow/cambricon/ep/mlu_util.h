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
#ifndef ONEFLOW_CAMBRICON_MLU_UTIL_H_
#define ONEFLOW_CAMBRICON_MLU_UTIL_H_

#include "oneflow/core/common/util.h"  // OF_DISALLOW_COPY_AND_MOVE
#include "cnrt.h"
#include "cnnl.h"
#include "cndev.h"
#include "cn_api.h"

namespace oneflow {

#define OF_MLU_CHECK(condition)                                                        \
  for (cnrtRet_t _cnrt_check_status = (condition); _cnrt_check_status != cnrtSuccess;) \
  LOG(FATAL) << "CNRT check failed: " #condition " : "                                 \
             << " (" << _cnrt_check_status << ") "

#define OF_CNNL_CHECK(condition)                                                                  \
  for (cnnlStatus_t _cnnl_check_status = (condition); _cnnl_check_status != CNNL_STATUS_SUCCESS;) \
  LOG(FATAL) << "CNNL check failed: " #condition " : "                                            \
             << " (error code:" << _cnnl_check_status                                             \
             << " " + std::string(cnnlErrorString(_cnnl_check_status)) + ") "

cnrtRet_t NumaAwareMluMallocHost(int32_t dev, void** ptr, size_t size);

class MluCurrentDeviceGuard final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MluCurrentDeviceGuard);
  explicit MluCurrentDeviceGuard(int32_t dev_id);
  MluCurrentDeviceGuard();
  ~MluCurrentDeviceGuard();

 private:
  int32_t saved_dev_id_ = -1;
};

int GetMluDeviceIndex();

int GetMluDeviceCount();

void SetMluDeviceIndex(int device_id);

void MluSynchronize(int device_id);

std::string cnnlErrorString(cnnlStatus_t status);

}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_MLU_UTIL_H_
