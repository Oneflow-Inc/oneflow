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
    num_recomputation_ = 0;
    num_eviction_ = 0;
    candidates_ = std::set<vm::DTREagerBlobObject*>();
    // candidates_ = std::unordered_set<vm::DTREagerBlobObject*>();
    // candidates_ = std::vector<vm::DTREagerBlobObject*>();
    start_time_ = std::chrono::steady_clock::now();
}

Maybe<vm::DTREagerBlobObject*> DTRTensorPool::find_best_tensor() {
    double min_cost = -1;
    vm::DTREagerBlobObject* best(nullptr);
    int tensor_id = 0;
    int evict_tensor_id = -1;
    if (oneflow::DTRDebugEnabled()) {
        std::cout << "Finding best tensor to evict..." << std::endl;
    }
    for (auto tensor : candidates_) {
        if (oneflow::DTRDebugEnabled()) {
            std::cout << "Is_in_memory: " << static_cast<bool>(tensor->is_in_memory()) << " " << "Is pinned: " << (tensor->num_pinned()) << ", Address: " << tensor << std::endl;
        }
        if (static_cast<bool>(tensor->compute_op()) && !tensor->is_pinned() && (tensor->is_evictable()) && tensor->is_in_memory()) {
            auto cur_cost = JUST(tensor->cost());
            if (min_cost < 0 || min_cost > cur_cost) {
                best = tensor;
                min_cost = cur_cost;
                evict_tensor_id = tensor_id;
            }
        }
        tensor_id++;
    }
    if (oneflow::DTRDebugEnabled()) {
        std::cout << "Evict " << evict_tensor_id << "th tensor." << std::endl;
    }
    num_eviction_++;
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
    if ((blob_object->is_evictable()) && (blob_object->memory() > thres)) {
        // for unordered_set version
        candidates_.insert(blob_object);
        // // for vector version
        // if (std::find(candidates_.begin(), candidates_.end(), blob_object) == candidates_.end()) {
        //     candidates_.emplace_back(blob_object);
        // }
    }
    return Maybe<void>::Ok();
}

Maybe<void> DTRTensorPool::evict(vm::DTREagerBlobObject* blob_object) {
    CHECK_NOTNULL_OR_RETURN(blob_object);
    // for unordered_set version
    if (candidates_.find(blob_object) != candidates_.end()) {
        candidates_.erase(blob_object);
        if (oneflow::DTRDebugEnabled()) {
            std::cout << "Successfully erase candidates before deconstruction." << std::endl;
        }
    }
    else {
        if (oneflow::DTRDebugEnabled()) {
            std::cout << "Unsuccessfully erase candidates before deconstruction." << std::endl;
        }
    }
    // for vector version
    // if (std::find(candidates_.begin(), candidates_.end(), blob_object) != candidates_.end()) {
    //     candidates_.erase(std::remove(candidates_.begin(), candidates_.end(), blob_object), candidates_.end());
    // }
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
    // std::cout << "===== Info of current tensor pool =====" << std::endl;
    std::cout << "Number of candidates: " << candidates_.size() << std::endl;
    size_t id = 0;
    for (const auto& candidate : candidates_) {
        std::cout << "id " << id++ << ", is_in_memory: " << candidate->is_in_memory() << ", input size: " << candidate->input_size() << ", is_evictable: " << candidate->is_evictable() << ", number of user_ops: " << candidate->num_user_ops() << ", address: " << candidate << std::endl;
    }
    // std::cout << "===== End info =====" << std::endl;
}

}   // namespace one
}   // namespace oneflow
