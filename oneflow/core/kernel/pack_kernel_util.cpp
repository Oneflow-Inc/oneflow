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
#include "oneflow/core/kernel/pack_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
void PackKernelUtil<device_type>::Pack(DeviceCtx* ctx, size_t in_index, size_t total_pack_num,
                                       const Blob* in_blob, Blob* out_blob) {
  size_t in_byte_size = in_blob->ByteSizeOfBlobBody();
  size_t out_byte_size = out_blob->ByteSizeOfBlobBody();
  CHECK_EQ(total_pack_num, out_byte_size / in_byte_size);

  const char* src_dptr = in_blob->dptr<char>();
  char* dst_dptr = out_blob->mut_dptr<char>() + in_byte_size * in_index;
  Memcpy<device_type>(ctx, dst_dptr, src_dptr, in_byte_size);
}

template<DeviceType device_type>
void PackKernelUtil<device_type>::Unpack(DeviceCtx* ctx, size_t out_index, size_t total_unpack_num,
                                         const Blob* in_blob, Blob* out_blob) {
  size_t in_byte_size = in_blob->ByteSizeOfBlobBody();
  size_t out_byte_size = out_blob->ByteSizeOfBlobBody();
  CHECK_EQ(total_unpack_num, in_byte_size / out_byte_size);

  const char* src_dptr = in_blob->dptr<char>() + out_byte_size * out_index;
  char* dst_dptr = out_blob->mut_dptr<char>();
  Memcpy<device_type>(ctx, dst_dptr, src_dptr, out_byte_size);
}

template class PackKernelUtil<DeviceType::kCPU>;
template class PackKernelUtil<DeviceType::kGPU>;

}  // namespace oneflow
