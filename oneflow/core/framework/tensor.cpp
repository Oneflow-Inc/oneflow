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
#include "oneflow/core/framework/tensor.h"

namespace oneflow {

namespace one {
Maybe<const compatible_py::Distribute> UndeterminedTensor::distribute() const {
  CHECK_OR_RETURN(distribute_) << Error::ValueError("Distribute is not determined.");
  return distribute_;
}

Maybe<const ParallelDesc> UndeterminedTensor::parallel_desc() const {
  CHECK_OR_RETURN(parallel_desc_) << Error::ValueError("Parallel_desc undetermined");
  return parallel_desc_;
}

Maybe<const Device> UndeterminedTensor::device() const {
  CHECK_OR_RETURN(device_) << Error::ValueError("Device undetermined.");
  return device_;
}

Maybe<DeterminedTensor> UndeterminedTensor::DetermineAndDestroySelf() {
  if (JUST(is_consistent())) {
    std::shared_ptr<ConsistentTensorImpl> impl;
    if (is_lazy()) {
      impl = std::make_shared<LazyConsistentTensorImpl>(shape(), dtype(), JUST(distribute()),
                                                        JUST(parallel_desc()));
    } else {
      impl = std::make_shared<EagerConsistentTensorImpl>(shape(), dtype(), JUST(distribute()),
                                                         JUST(parallel_desc()));
    }
    impl->set_requires_grad(requires_grad());
    impl->set_retain_grad(retain_grad());
    return std::static_pointer_cast<DeterminedTensor>(std::make_shared<ConsistentTensor>(impl));
  } else {
    std::shared_ptr<MirroredTensorImpl> impl;
    if (is_lazy()) {
      impl = std::make_shared<LazyMirroredTensorImpl>(shape(), dtype(), JUST(device()));
    } else {
      impl = std::make_shared<EagerMirroredTensorImpl>(shape(), dtype(), JUST(device()));
    }
    impl->set_requires_grad(requires_grad());
    impl->set_retain_grad(retain_grad());
    return std::static_pointer_cast<DeterminedTensor>(std::make_shared<MirroredTensor>(impl));
  }
}

bool UndeterminedTensor::is_leaf() const { TODO(); }

}  // namespace one
}  // namespace oneflow
