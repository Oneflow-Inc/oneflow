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
#include "oneflow/core/framework/eager_blob_object_util.h"
#include "oneflow/core/eager/eager_blob_object.h"

namespace oneflow {

Maybe<bool> ProducedAndLastUsedOnSameStream(const std::shared_ptr<vm::EagerBlobObject>& lhs,
                                            const std::shared_ptr<vm::EagerBlobObject>& rhs) {
  return JUST(lhs->producer_stream()) == JUST(rhs->producer_stream())
         && JUST(lhs->last_used_stream()) == JUST(rhs->last_used_stream());
}

}  // namespace oneflow
