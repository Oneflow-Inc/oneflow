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

#include "oneflow/core/eager/local_call_opkernel_phy_instr_operand.h"
#include "oneflow/core/framework/tensor_pool.h"

namespace oneflow {

namespace vm {
class LocalCallOpKernelPhyInstrOperand;
}

namespace one {

DTRTensorPool::DTRTensorPool() {
  num_recomputation_ = 0;
  num_eviction_ = 0;
  num_destruction_ = 0;
  // candidates_ = std::set<std::weak_ptr<vm::DTREagerBlobObject>>();
  candidates_ = std::vector<std::weak_ptr<vm::DTREagerBlobObject>>();
  start_time_ = std::chrono::steady_clock::now();
}

Maybe<vm::DTREagerBlobObject*> DTRTensorPool::find_best_tensor() {
  double min_cost = -1;
  vm::DTREagerBlobObject* best(nullptr);
  int tensor_id = 0;
  int evict_object_id = -1;
  if (oneflow::DTRDebugEnabled()) { std::cout << "Finding best tensor to evict..." << std::endl; }
  int id = 0;
  for (const auto& object : candidates_) {
    if (auto shared_object = object.lock()) {
      // if (oneflow::DTRDebugEnabled()) {
      // std::cout << "id " << id++ << ", is_in_memory: " <<
      // static_cast<bool>(tensor->is_in_memory()) << " " << "is pinned: " << (tensor->num_pinned())
      // << ", Address: " << static_cast<const void *>(tensor->object_dptr()) << ", shape: " <<
      // tensor->mut_blob_desc()->shape() << ", data_type: " << tensor->mut_blob_desc()->data_type()
      // << ", body_bytes: " << tensor->BlobBodyBytes() << std::endl; std::cout << "id " << id++ <<
      // ", is_in_memory: " << static_cast<bool>(tensor->is_in_memory()) << " " << "is pinned: " <<
      // (tensor->num_pinned()) << ", Address: " << tensor << ", shape: " <<
      // tensor->mut_blob_desc()->shape() << ", data_type: " << tensor->mut_blob_desc()->data_type()
      // << ", body_bytes: " << tensor->BlobBodyBytes() << std::endl;
      // }
      if (static_cast<bool>(shared_object->compute_op()) && !shared_object->is_pinned()
          && (shared_object->is_evictable()) && shared_object->is_in_memory()) {
        auto cur_cost = JUST(shared_object->cost());
        if (min_cost < 0 || min_cost > cur_cost) {
          best = shared_object.get();
          min_cost = cur_cost;
          evict_object_id = tensor_id;
        }
      }
      tensor_id++;
    } else {
      JUST(display());
      std::cout << "Unable to lock candidates in tensor pool: " << id
                << ", is_expired: " << object.expired() << std::endl;
    }
    id++;
  }
  if (oneflow::DTRDebugEnabled()) {
    std::cout << "Evict " << evict_object_id << "th object." << std::endl;
  }
  num_eviction_++;
  return best;
}

Maybe<bool> DTRTensorPool::find_best_tensor_and_evict() {
  auto* best = JUST(find_best_tensor());
  CHECK_NOTNULL_OR_RETURN(best);
  JUST(best->evict());
  return true;
}

Maybe<void> DTRTensorPool::insert(std::shared_ptr<vm::DTREagerBlobObject> blob_object,
                                  size_t thres) {
  CHECK_NOTNULL_OR_RETURN(blob_object);
  if ((blob_object->is_evictable()) && (blob_object->memory() > thres)) {
    // // for set version
    // candidates_.insert(blob_object);

    // for vector version
    for (const auto& object : candidates_) {
      if (object.lock() == blob_object) { return Maybe<void>::Ok(); }
    }

    candidates_.emplace_back(blob_object);
  }
  return Maybe<void>::Ok();
}

Maybe<void> DTRTensorPool::evict(vm::DTREagerBlobObject* blob_object) {
  CHECK_NOTNULL_OR_RETURN(blob_object);
  auto object = candidates_.begin();
  while (object != candidates_.end()) {
    // std::cout << "object->lock.get(): " << object->lock().get() << std::endl;
    if (object->lock().get() == blob_object) {
      if (oneflow::DTRDebugEnabled()) {
        std::cout << "Successfully erase candidates before deconstruction: " << blob_object
                  << std::endl;
      }
      candidates_.erase(object);
      num_destruction_++;
      return Maybe<void>::Ok();
    }
    ++object;
  }

  if (oneflow::DTRDebugEnabled()) {
    std::cout << "Unsuccessfully erase candidates before deconstruction: " << blob_object
              << std::endl;
  }

  return Maybe<void>::Ok();
}

Maybe<void> DTRTensorPool::clear() {
  auto object = candidates_.begin();
  while (object != candidates_.end()) {
    if (object->lock().get() == nullptr) {
      if (oneflow::DTRDebugEnabled()) {
        std::cout << "Erase nullptr candidates from tensor_pool." << std::endl;
      }
      candidates_.erase(object);
      num_destruction_++;
      return Maybe<void>::Ok();
    }
    ++object;
  }
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

Maybe<void> DTRTensorPool::display() {
  std::cout << "===== Info of current tensor pool =====" << std::endl;
  std::cout << "Number of candidates: " << candidates_.size() << std::endl;
  size_t id = 0;
  for (const auto& candidate : candidates_) {
    if (auto wp = candidate.lock()) {
      auto tmp = std::dynamic_pointer_cast<vm::DTREagerBlobObject>(wp);
      CHECK_NOTNULL_OR_RETURN(tmp);
      std::cout << "id " << id << ", is_in_memory: " << tmp->is_in_memory()
                << ", input size: " << tmp->input_size()
                << ", is_evictable: " << tmp->is_evictable()
                << ", number of user_ops: " << tmp->num_user_ops() << ", address: " << tmp
                << ", nullptr? " << (tmp == nullptr) << std::endl;
    }
    // std::cout << "id " << id++ << ", is_in_memory: " << candidate->is_in_memory() << ", address:
    // " << candidate << std::endl;
    id++;
  }
  // for (const auto& candidate : candidates_) {
  //     const auto* tmp = dynamic_cast<vm::DTREagerBlobObject*>(candidate);
  //     std::cout << "Input info--------------- " << std::endl;
  //     size_t iid = 0;
  //     const auto* ptr =
  //     dynamic_cast<vm::LocalCallOpKernelPhyInstrOperand*>(candidate->compute_op());
  //     CHECK_NOTNULL_OR_RETURN(ptr);
  //     for (const auto& input : *ptr->inputs()) {
  //         CHECK_OR_RETURN(static_cast<bool>(input.get()));
  //         const auto* dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(input.get());
  //         CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
  //         std::cout << "Input id: " << iid++ << ", address: " << dtr_blob_object << ", ref_cnt: "
  //         << input.use_count() << std::endl;
  //     }

  //     // std::cout << "Output info--------------- " << std::endl;
  //     // for (int i = 0; i < candidate->num_user_ops(); ++i) {
  //     //     size_t oid = 0;
  //     //     std::cout << "The " << i << "th user_op: " << std::endl;
  //     //     const auto* ptr =
  //     dynamic_cast<vm::LocalCallOpKernelPhyInstrOperand*>(CHECK_JUST(candidate->user_op(i)));
  //     //     for (const auto& output: *ptr->outputs()) {
  //     //         CHECK_OR_RETURN(static_cast<bool>(output.get()));
  //     //         const auto* dtr_blob_object =
  //     dynamic_cast<vm::DTREagerBlobObject*>(output.get());
  //     //         CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
  //     //         std::cout << "Output id: " << oid++ << ", address: " << dtr_blob_object << ",
  //     ref_cnt: " << output.use_count() << std::endl;
  //     //     }
  //     // }
  //     // std::cout << "id " << id++ << ", is_in_memory: " << candidate->is_in_memory() << ",
  //     input size: " << candidate->input_size() << ", is_evictable: " << candidate->is_evictable()
  //     << ", number of user_ops: " << candidate->num_user_ops() << ", address: " <<
  //     static_cast<const void *>(candidate->object_dptr()) << ", nullptr? " << (tmp == nullptr) <<
  //     std::endl;
  //     // std::cout << "id " << id++ << ", is_in_memory: " << candidate->is_in_memory() << ",
  //     input size: " << candidate->input_size() << ", is_evictable: " << candidate->is_evictable()
  //     << ", number of user_ops: " << candidate->num_user_ops() << ", address: " << candidate <<
  //     ", nullptr? " << (tmp == nullptr) << std::endl;
  // }
  // std::cout << "===== End info =====" << std::endl;
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
