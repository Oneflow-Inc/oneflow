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
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/user/kernels/roll_kernel.h"

namespace oneflow {

template<typename T>
struct RollChange<DeviceType::kCPU, T> {
     static void Invoke(DeviceCtx *ctx, bool isPositive, int32_t cpyFirst, 
                       int32_t cpySec, const T *x, T *y){

     if(isPositive) {
         memcpy(y, x+cpySec, cpyFirst*sizeof(T));
         memcpy(y+cpyFirst, x, cpySec*sizeof(T));     
     } else {
         memcpy(y, x+cpyFirst, cpySec*sizeof(T));
         memcpy(y+cpySec, x, cpyFirst*sizeof(T));
     }
     // std::cout << "shift=" << shift << " width=" << width << " len="  << len << std::endl; 
     }
};    


REGISTER_ROLL_USER_KERNEL(DeviceType::kCPU, float)
REGISTER_ROLL_USER_KERNEL(DeviceType::kCPU, double)

}   // namespace oneflow
