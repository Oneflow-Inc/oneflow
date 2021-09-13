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

#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_POOL_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_POOL_H_

#include <unordered_set>
#include <chrono>

#include "oneflow/core/eager/eager_blob_object.h"

namespace oneflow {
namespace one {

struct DTRTensorPool {
    DTRTensorPool();

    Maybe<void> insert(vm::DTREagerBlobObject* blob_object, size_t thres=0);
    Maybe<void> evict(vm::DTREagerBlobObject* blob_object);

    Maybe<vm::DTREagerBlobObject*> find_best_tensor();
    Maybe<void> find_best_tensor_and_evict();
    // do not need Maybe
    const std::chrono::steady_clock::time_point start_time() { return start_time_; }
    double duration();
    void display();

    // TODO: Implementation of disjoint-set data structure

private:
    // shared_ptr or not?
    // downcast to DTREagerBlobObject*, could use unique interfaces
    // At first, we use unordered_set for efficiency. Now use vector to view id of candidates in order.
    // std::unordered_set<vm::DTREagerBlobObject*> candidates_;
    std::vector<vm::DTREagerBlobObject*> candidates_;
    std::chrono::steady_clock::time_point start_time_;

};

}   // namespace one
}   // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_POOL_H_
