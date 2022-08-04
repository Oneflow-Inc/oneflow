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
#include "oneflow/core/eager/eager_blob_object.h"

namespace oneflow {

class Device;
class TensorTuple;
class ParallelDesc;

namespace one {

class Tensor;

Maybe<Tensor> Broadcast(const std::shared_ptr<Tensor>& tensor, int64_t src_rank,
                        Symbol<ParallelDesc> parallel_desc, bool inplace);

Maybe<TensorTuple> Broadcast(const TensorTuple& inputs, int64_t src_rank,
                             Symbol<ParallelDesc> parallel_desc, bool inplace);

}  // namespace one
}  // namespace oneflow
