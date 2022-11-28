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
#ifndef ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"

namespace oneflow {

namespace ep {

class Stream;

}

template<DeviceType device_type>
void Memcpy(ep::Stream* stream, void* dst, const void* src, size_t sz) {
  CHECK_EQ(device_type, stream->device_type()) << "Device type mismatch";
  std::unique_ptr<ep::primitive::Memcpy> primitive =
      ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(stream->device_type(),
                                                                ep::primitive::MemcpyKind::kDtoD);
  CHECK(primitive) << "Can not create Memcpy primitive for device type " << device_type;
  primitive->Launch(stream, dst, src, sz);
}

template<DeviceType device_type>
void Memset(ep::Stream* stream, void* dst, const char value, size_t sz) {
  CHECK_EQ(device_type, stream->device_type()) << "Device type mismatch";
  std::unique_ptr<ep::primitive::Memset> primitive =
      ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(stream->device_type());
  CHECK(primitive) << "Can not create Memset primitive for device type " << device_type;
  primitive->Launch(stream, dst, value, sz);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_
