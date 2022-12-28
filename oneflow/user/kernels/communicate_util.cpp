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
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/user/kernels/collective_communication/include/send.h"
#include "oneflow/user/kernels/collective_communication/include/recv.h"

namespace oneflow {

namespace {

const void** ThreadLocalSrcDataPtr() {
  static thread_local const void* data_ptr = nullptr;
  return &data_ptr;
}

}  // namespace

bool IsSendAndRecvRegistered(DeviceType device_type) {
  return ccl::IsSendRegistered(device_type) && ccl::IsRecvRegistered(device_type);
}

Maybe<void> Send(const void* in, size_t elem_cnt, DataType dtype, int64_t dst,
                 DeviceType device_type, ep::Stream* stream) {
  if (GlobalProcessCtx::Rank() == dst) {
    auto** src_data_ptr = ThreadLocalSrcDataPtr();
    CHECK_OR_RETURN(*src_data_ptr == nullptr);
    *src_data_ptr = in;
  } else {
    std::unique_ptr<ccl::Send> send =
        ccl::NewCollectiveCommunication<ccl::Send>(device_type, dtype);
    send->Launch(stream, in, elem_cnt, dst);
  }
  return Maybe<void>::Ok();
}

Maybe<void> Recv(void* out, size_t elem_cnt, DataType dtype, int64_t src, DeviceType device_type,
                 ep::Stream* stream) {
  if (GlobalProcessCtx::Rank() == src) {
    size_t buffer_size = elem_cnt * GetSizeOfDataType(dtype);
    auto** src_data_ptr = ThreadLocalSrcDataPtr();
    const void* in = *src_data_ptr;
    CHECK_OR_RETURN(*src_data_ptr != nullptr);
    std::unique_ptr<ep::primitive::Memcpy> memcpy_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(device_type,
                                                                  ep::primitive::MemcpyKind::kDtoD);
    CHECK(memcpy_primitive) << "Can not create Memcpy primitive for device type " << device_type;
    memcpy_primitive->Launch(stream, out, in, buffer_size);
    *src_data_ptr = nullptr;
  } else {
    std::unique_ptr<ccl::Recv> recv =
        ccl::NewCollectiveCommunication<ccl::Recv>(device_type, dtype);
    recv->Launch(stream, out, elem_cnt, src);
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
