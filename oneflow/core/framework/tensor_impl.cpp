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
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {

namespace one {

namespace {

void SetParallelDescByDevice(const std::shared_ptr<const Device>& device, ParallelDesc* parallel_desc) {
}

void SetDeviceByParallelDesc(const std::shared_ptr<const ParallelDesc>& parallel_desc, Device* device) {
}

}

LazyMirroredTensorImpl::LazyMirroredTensorImpl(const std::shared_ptr<const Shape>& shape, DataType dtype,
                         const std::shared_ptr<const Device>& device)
  : shape_(shape), dtype_(dtype), device_(device) {
  SetParallelDescByDevice(device, mut_parallel_desc());
}

EagerMirroredTensorImpl::EagerMirroredTensorImpl(const std::shared_ptr<const Shape>& shape, DataType dtype,
                          const std::shared_ptr<const Device>& device)
  : shape_(shape), dtype_(dtype), device_(device) {
  SetParallelDescByDevice(device, mut_parallel_desc());
}

// Support later
/* void LazyMirroredTensorImpl::set_parallel_desc(const std::shared_ptr<const ParallelDesc>& parallel_desc) { */
/*   parallel_desc_ = parallel_desc; */
/* } */

/* void EagerMirroredTensorImpl::set_parallel_desc(const std::shared_ptr<const ParallelDesc>& parallel_desc) { */
/*   parallel_desc_ = parallel_desc; */
/* } */

/* void LazyMirroredTensorImpl::set_device(const std::shared_ptr<const Device>& device) { */ 
/*   device_ = device; */ 
/* } */

}

}  // namespace oneflow
