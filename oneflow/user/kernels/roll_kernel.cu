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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/framework/util.h"
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/user/kernels/roll_kernel.h"

namespace oneflow {
    
template<typename T>
struct RollChange<DeviceType::kGPU, T> {
    static void Invoke(DeviceCtx *ctx, bool isPositive, int32_t cpyFirst, 
                       int32_t cpySec, const T *x, T *y) {
     if(isPositive) {
        cudaMemcpy(y, x+cpySec, cpyFirst*sizeof(T), cudaMemcpyDefault);
        cudaMemcpy(y+cpyFirst, x, cpySec*sizeof(T), cudaMemcpyDefault);     
     } else {
        cudaMemcpy(y, x+cpyFirst, cpySec*sizeof(T), cudaMemcpyDefault);
        cudaMemcpy(y+cpySec, x, cpyFirst*sizeof(T), cudaMemcpyDefault);
     }
    } 
};

REGISTER_ROLL_USER_KERNEL(DeviceType::kGPU, float);
REGISTER_ROLL_USER_KERNEL(DeviceType::kGPU, double);

} // namespace oneflow
