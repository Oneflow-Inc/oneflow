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

#include "oneflow/core/framework/tensor_pool.h"

namespace oneflow {
namespace one {

DTRTensorPool::DTRTensorPool() {
    start_time_ = std::chrono::steady_clock::now();
}

Maybe<vm::DTREagerBlobObject*> DTRTensorPool::find_best_tensor() {
    double min_cost = -1;
    vm::DTREagerBlobObject* best(nullptr);
    int tensor_id = 0;
    std::cout << candidates_.size() << std::endl;
    for (auto tensor : candidates_) {
        tensor_id++;
        std::cout << static_cast<bool>(tensor->compute_op()) << " " << (!tensor->is_pinned()) << std::endl;
        if (static_cast<bool>(tensor->compute_op()) && !tensor->is_pinned() && (tensor->input_size()>0)) {
            auto cur_cost = JUST(tensor->cost());
            if (min_cost < 0 || min_cost > cur_cost) {
                best = tensor;
                min_cost = cur_cost;
            }
        }
    }
    return best;
}

Maybe<void> DTRTensorPool::find_best_tensor_and_evict() {
    auto* best = JUST(find_best_tensor());
    CHECK_NOTNULL_OR_RETURN(best);
    JUST(best->evict());
    return Maybe<void>::Ok();
}

Maybe<void> DTRTensorPool::insert(vm::DTREagerBlobObject* blob_object, size_t thres) {
    CHECK_NOTNULL_OR_RETURN(blob_object);
    if ((blob_object->input_size() > 0) && (blob_object->memory() > thres)) {
        candidates_.insert(blob_object);
    }
    return Maybe<void>::Ok();
}

Maybe<void> DTRTensorPool::evict(vm::DTREagerBlobObject* blob_object) {
    CHECK_NOTNULL_OR_RETURN(blob_object);
    candidates_.erase(blob_object);
    return Maybe<void>::Ok();
}

double DTRTensorPool::duration() {
    auto t2 = std::chrono::steady_clock::now();
    // time in seconds
    std::chrono::duration<double> time_span = t2 - start_time_;
    // // time in milli
    // std::chrono::duration<double ,std::milli> time_span = t2 - start_time_;
    return time_span.count();
}

void DTRTensorPool::display() {
    std::cout << "Info of current tensor pool:" << std::endl;
    std::cout << "Number of candidates: " << candidates_.size() << std::endl;
    size_t id = 0;
    for (const auto& candidate : candidates_) {
        std::cout << "id " << id++ << ", is_in_memory: " << candidate->is_in_memory() << ", input size: " << candidate->input_size() << std::endl;
    }
}

}   // namespace one
}   // namespace oneflow
