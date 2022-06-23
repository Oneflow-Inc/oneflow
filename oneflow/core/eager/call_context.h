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
#ifndef ONEFLOW_CORE_EAGER_CALL_CONTEXT_H_
#define ONEFLOW_CORE_EAGER_CALL_CONTEXT_H_

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/op_interpreter.h"

namespace oneflow {

namespace one {

class ConsistentTensorInferResult;

using EagerBlobObjectList = std::vector<std::shared_ptr<vm::EagerBlobObject>>;
using EagerBlobObjectListPtr =
    std::shared_ptr<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>;

}  // namespace one

namespace eager {

struct CallContext {
  ComposedAttrMap composed_attrs;
  one::EagerBlobObjectListPtr inputs;
  one::EagerBlobObjectListPtr outputs;
  std::shared_ptr<const one::ConsistentTensorInferResult> consistent_tensor_infer_result;
  const one::OpExprInterpContext op_interp_ctx;
};

}  // namespace eager

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_CALL_CONTEXT_H_
