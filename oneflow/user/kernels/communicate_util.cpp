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
#include "oneflow/user/kernels/communicate_util.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {

namespace {

const void** ThreadLocalSrcDataPtr() {
  static thread_local const void* data_ptr = nullptr;
  return &data_ptr;
}

}  // namespace

template<DeviceType device_type>
Maybe<void> Send(const void* in, size_t elem_cnt, DataType dtype, int64_t dst, DeviceCtx* ctx) {
  if (GlobalProcessCtx::Rank() == dst) {
    auto** src_data_ptr = ThreadLocalSrcDataPtr();
    CHECK_OR_RETURN(*src_data_ptr == nullptr);
    *src_data_ptr = in;
  } else {
    JUST(ccl::Send<device_type>(in, elem_cnt, dtype, dst, ctx));
  }
  return Maybe<void>::Ok();
}

template<DeviceType device_type>
Maybe<void> Recv(void* out, size_t elem_cnt, DataType dtype, int64_t src, DeviceCtx* ctx) {
  if (GlobalProcessCtx::Rank() == src) {
    size_t buffer_size = elem_cnt * GetSizeOfDataType(dtype);
    auto** src_data_ptr = ThreadLocalSrcDataPtr();
    const void* in = *src_data_ptr;
    CHECK_OR_RETURN(*src_data_ptr != nullptr);
    Memcpy<device_type>(ctx, out, in, buffer_size);
    *src_data_ptr = nullptr;
  } else {
    JUST(ccl::Recv<device_type>(out, elem_cnt, dtype, src, ctx));
  }
  return Maybe<void>::Ok();
}

template Maybe<void> Send<DeviceType::kCPU>(const void* in, size_t elem_cnt, DataType dtype,
                                            int64_t dst, DeviceCtx* ctx);

template Maybe<void> Recv<DeviceType::kCPU>(void* out, size_t elem_cnt, DataType dtype, int64_t src,
                                            DeviceCtx* ctx);

#if defined(WITH_CUDA) && HAS_GPU_SEND_RECV
template Maybe<void> Send<DeviceType::kGPU>(const void* in, size_t elem_cnt, DataType dtype,
                                            int64_t dst, DeviceCtx* ctx);

template Maybe<void> Recv<DeviceType::kGPU>(void* out, size_t elem_cnt, DataType dtype, int64_t src,
                                            DeviceCtx* ctx);
#endif
}  // namespace oneflow
