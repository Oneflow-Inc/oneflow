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
#include "oneflow/core/job/global_for.h"

namespace oneflow {

namespace one {
std::shared_ptr<DeterminedTensor> UndeterminedTensor::DetermineAndDestroySelf() {
  bool is_lazy = *Global<bool, EagerExecution>::Get();
  if (is_consistent()) {
    std::shared_ptr<ConsistentTensorImpl> impl;
    if (is_lazy) {
      impl = std::make_shared<LazyConsistentTensorImpl>(shape(), dtype(), distribute(),
                                                        parallel_desc());
    } else {
      impl = std::make_shared<EagerConsistentTensorImpl>(shape(), dtype(), distribute(),
                                                         parallel_desc());
    }
    return std::static_pointer_cast<DeterminedTensor>(std::make_shared<ConsistentTensor>(impl));
  } else {
    std::shared_ptr<MirroredTensorImpl> impl;
    if (is_lazy) {
      impl = std::make_shared<LazyMirroredTensorImpl>(shape(), dtype(), device());
    } else {
      impl = std::make_shared<EagerMirroredTensorImpl>(shape(), dtype(), device());
    }
    return std::static_pointer_cast<DeterminedTensor>(std::make_shared<MirroredTensor>(impl));
  }
}
}  // namespace one
}  // namespace oneflow
