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
#include <set>
#include <chrono>

#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/eager/eager_blob_object.h"

namespace oneflow {
namespace one {

struct DTRTensorPool {
    DTRTensorPool();
    ~DTRTensorPool() { 
        std::cout << "=======================" << std::endl;
        std::cout << "Destruct DTRTensorPool." << std::endl;
        std::cout << "Times of eviction: " << num_eviction_ << std::endl;
        std::cout << "Times of recomputation: " << num_recomputation_ << std::endl;
        std::cout << "Times of destruction: " << num_destruction_ << std::endl;
        if (oneflow::DTRDebugEnabled()) {
            display();
        }
    }

    Maybe<void> insert(std::shared_ptr<vm::DTREagerBlobObject> blob_object, size_t thres=0);
    Maybe<void> evict(vm::DTREagerBlobObject* blob_object);

    Maybe<vm::DTREagerBlobObject*> find_best_tensor();
    Maybe<bool> find_best_tensor_and_evict();
    // do not need Maybe
    const std::chrono::steady_clock::time_point start_time() { return start_time_; }
    double duration();
    Maybe<void> display();
    void add_recompute_times() { num_recomputation_++; }

    // TODO: Implementation of disjoint-set data structure

private:
    // vector for eviction, set for non-eviction.
    // std::set<std::weak_ptr<vm::DTREagerBlobObject>> candidates_;
    std::vector<std::weak_ptr<vm::DTREagerBlobObject>> candidates_;
    std::chrono::steady_clock::time_point start_time_;
    int num_eviction_;
    int num_recomputation_;
    int num_destruction_;
};

}   // namespace one
}   // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_POOL_H_
