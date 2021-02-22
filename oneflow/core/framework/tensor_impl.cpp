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
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {

namespace one {

bool Tensor::is_lazy() const { return !EagerExecutionEnabled(); }

MirroredTensor::MirroredTensor(const std::shared_ptr<Shape>& shape, DataType dtype,
                               const std::shared_ptr<Device>& device) {
  if (is_lazy()) {
    impl_ = std::make_shared<MirroredLazyTensorImpl>(shape, dtype, device);
  } else {
    impl_ = std::make_shared<MirroredEagerTensorImpl>(shape, dtype, device);
  }
}

ConsistentTensor::ConsistentTensor(const std::shared_ptr<Shape>& shape, DataType dtype,
                                   const std::shared_ptr<compatible_py::Distribute>& distribute,
                                   std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  if (is_lazy()) {
    impl_ = std::make_shared<ConsistentLazyTensorImpl>(shape, dtype, distribute, parallel_conf);
  } else {
    impl_ = std::make_shared<ConsistentEagerTensorImpl>(shape, dtype, distribute, parallel_conf);
  }
}

int64_t EagerTensorImpl::numpy_size() const {
  return blob_object()->parallel_desc_symbol()->parallel_num();
}

int64_t EagerTensorImpl::numpy_list_size() const {
  return blob_object()->parallel_desc_symbol()->parallel_num();
}

}  // namespace one

}  // namespace oneflow
