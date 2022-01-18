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
#include "oneflow/user/kernels/stateful_local_opkernel.h"

namespace oneflow {

namespace vm {
class LocalCallOpKernelPhyInstrOperand;
}

namespace one {

DTRTensorPool::DTRTensorPool() : duration_(0), total_memory_bytes_(0), num_eviction_(0), num_recomputation_(0), num_destruction_(0) {
  start_time_ = std::chrono::steady_clock::now();
}

namespace {
void printInfo(const std::shared_ptr<vm::DTREagerBlobObject>& debo) {
  double mem = debo->BlobBodyBytes() * 1. / 1024 / 1024;
  const std::string& compute_op =
      debo->compute_op()->shared_opkernel()->user_op_conf_->op_type_name();
  std::cout << "is_in_memory: " << static_cast<bool>(debo->is_in_memory())
            << ", is pinned: " << (debo->num_pinned())
            << ", is evictable: " << (debo->is_evictable()) << ", compute op: " << compute_op
            << ", shape: " << debo->mut_blob_desc()->shape() << ", memory: " << mem << "MB"
            << ", ebo address: " << debo.get() << ", blob dptr: "
            << debo->blob().dptr()
            // << ", parent_depth: " << debo->parent_depth()
            // << ", child_depth: " << debo->child_depth()
            << ", tensor buffer dptr: "
            << static_cast<const void*>(debo->tensor_buffer()->blob_dptr())
            // << ", data_type: " << debo->mut_blob_desc()->data_type()
            << std::endl;
}
}  // namespace

void DTRTensorPool::inc_num_eviction() { num_eviction_++; }

Maybe<vm::DTREagerBlobObject*> DTRTensorPool::find_best_tensor() {
  double min_cost = -1;
  vm::DTREagerBlobObject* best(nullptr);
  int evict_object_id = -1;
  if (oneflow::DTRDebugEnabled()) { std::cout << "Finding best tensor to evict..." << std::endl; }
  int id = 0;
  double tensor_pool_mem = 0.;
  for (const auto& object : candidates_) {
    if (auto shared_object = object.lock()) {
      if (static_cast<bool>(shared_object->compute_op()) && !shared_object->is_pinned()
          && (shared_object->is_evictable()) && shared_object->is_in_memory()) {
        const double cur_cost = JUST(shared_object->cost());
        // const double cur_cost = JUST(shared_object->reverse_cost());
        if (min_cost < 0 || min_cost > cur_cost) {
          best = shared_object.get();
          min_cost = cur_cost;
          evict_object_id = id;
        }
      }
      if (static_cast<bool>(shared_object->compute_op())) {
        if (oneflow::DTRDebugLevel() >= 2) {
          std::cout << "id " << id << ", ";
          printInfo(shared_object);
        }
      }
      if (oneflow::DTRDebugEnabled()) {
        // copy op in lenet/alexnet model is always to copy parameters
        // from cpu to gpu during model.to('cuda')
        // copying from cpu to gpu uses cuda h2d memory pool, unrelated
        // to the cuda memory pool dtr uses.
        double mem = shared_object->BlobBodyBytes() * 1. / 1024 / 1024;
        const std::string& compute_op =
            shared_object->compute_op()->shared_opkernel()->user_op_conf_->op_type_name();
        if (shared_object->is_in_memory() && compute_op != "copy") { tensor_pool_mem += mem; }
      }
    } else {
      JUST(display());
      std::cout << "Unable to lock candidates in tensor pool: " << id
                << ", is_expired: " << object.expired() << std::endl;
    }
    id++;
  }
  if (oneflow::DTRDebugEnabled()) {
    std::cout << "pool mem is " << tensor_pool_mem << "MB" << std::endl;
    if (best != nullptr) {
      std::cout << "Evict " << evict_object_id << "th object , cost is " << min_cost
                << ", compute op is "
                << (best ? best->compute_op()->shared_opkernel()->op_type_name() : "null")
                << ", addr is " << best << ", memory is "
                << best->BlobBodyBytes() * 1. / 1024 / 1024 << std::endl;
      // const auto pd = best->parent_depth();
      // const auto cd = best->child_depth();
      // std::cout << "depth is " << pd << "+" << cd << "=" << (pd + cd) << std::endl;
    }
  }
  num_eviction_++;
  // if (min_cost > 10) {
  // JUST(display2());
  // exit(1);
  // }
  return best;
}

Maybe<bool> DTRTensorPool::find_best_tensor_and_evict() {
  auto* best = JUST(find_best_tensor());
  if (best == nullptr) { return false; }
  JUST(best->evict());
  JUST(update_after_evict(best));
  return true;
}

Maybe<void> DTRTensorPool::insert(std::shared_ptr<vm::DTREagerBlobObject> blob_object,
                                  size_t thres) {
  CHECK_NOTNULL_OR_RETURN(blob_object);
  CHECK_EQ_OR_RETURN(thres, 0);
  // if ((blob_object->is_evictable()) && (blob_object->memory() > thres)) {
  // if (true) {
  if (blob_object->compute_op()->shared_opkernel()->op_type_name() != "copy"
      && blob_object->blob().mem_case().has_device_cuda_mem()) {
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

Maybe<void> DTRTensorPool::clear() {
  auto object = candidates_.begin();
  while (object != candidates_.end()) {
    if (object->lock().get() == nullptr) {
      if (oneflow::DTRDebugEnabled()) {
        // std::cout << "Erase nullptr candidates from tensor_pool." << std::endl;
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
  return duration_;
  // auto t2 = std::chrono::steady_clock::now();
  // // time in seconds
  // std::chrono::duration<double> time_span = t2 - start_time_;
  // // // time in milli
  // // std::chrono::duration<double ,std::milli> time_span = t2 - start_time_;
  // return time_span.count();
}

void DTRTensorPool::time_flies(double t) {
  duration_ += t;
  if (oneflow::DTRDebugEnabled()) { LOG(INFO) << "time flies " << t << " to " << duration_; }
}

Maybe<void> DTRTensorPool::display2() {
  std::cout << "===== Info of current tensor pool =====" << std::endl;
  std::cout << "Number of candidates: " << candidates_.size() << std::endl;
  int id = 0;
  float total_mem = 0;
  float living_mem = 0;
  float pinned_mem = 0;
  for (const auto& object : candidates_) {
    if (auto shared_object = object.lock()) {
      // if (shared_object->is_in_memory()) {
      if (true) {
        std::cout << "id " << id << ", ";
        printInfo(shared_object);
        total_mem += shared_object->BlobBodyBytes() * 1. / 1024 / 1024;
        if (shared_object->is_in_memory()) {
          living_mem += shared_object->BlobBodyBytes() * 1. / 1024 / 1024;
          if (shared_object->is_pinned()) {
            pinned_mem += shared_object->BlobBodyBytes() * 1. / 1024 / 1024;
          }
        }
      }
    }
    id++;
  }
  std::cout << "Total memory: " << total_mem << "MB" << std::endl;
  std::cout << "Living memory: " << living_mem << "MB" << std::endl;
  std::cout << "Pinned memory: " << pinned_mem << "MB" << std::endl;
  return Maybe<void>::Ok();
}

Maybe<void> DTRTensorPool::display() {
  std::cout << "===== Info of current tensor pool =====" << std::endl;
  std::cout << "Number of candidates: " << candidates_.size() << std::endl;
  return Maybe<void>::Ok();
}

// Disjoint Node Set
void DTRTensorPool::merge(std::shared_ptr<vm::DisjNode>& x, std::shared_ptr<vm::DisjNode>& y) {
  auto&& parent_x = find_father(x);
  auto&& parent_y = find_father(y);
  if (parent_x.get() == parent_y.get()) { return; }

  parent_y->set_compute_time(parent_y->compute_time() + parent_x->compute_time());
  parent_x->set_parent(parent_y);

  y->reset_pesudo_node();
  x->reset_pesudo_node();
}

void DTRTensorPool::pesudo_merge(std::shared_ptr<vm::DisjNode>& x, std::shared_ptr<vm::DisjNode>& y) {
  auto pesudo_x = x->pesudo_node();
  auto pesudo_y = y->pesudo_node();
  auto&& pesudo_parent_x = find_father(pesudo_x);
  auto&& pesudo_parent_y = find_father(pesudo_y);
  if (pesudo_parent_x.get() == pesudo_parent_y.get()) { return; }

  pesudo_parent_y->set_compute_time(pesudo_parent_y->compute_time() + pesudo_parent_x->compute_time());
  pesudo_parent_x->set_parent(pesudo_parent_y);
  pesudo_parent_y->add_cnt();
  pesudo_parent_x->set_cnt(pesudo_parent_y->cnt());
}

std::shared_ptr<vm::DisjNode> DTRTensorPool::find_father(std::shared_ptr<vm::DisjNode>& x) {
  if (x->is_root()) {
    return x;
  } else {
    auto fa = x->parent();
    auto&& y = find_father(fa);
    x->set_parent(y);
    return y;
  }
}

void DTRTensorPool::update_after_compute(vm::DTREagerBlobObject* obj) {
  auto&& fa = find_father(obj->node);
  fa->set_compute_time(fa->compute_time() - obj->node->compute_time());
  obj->reset_node(obj->compute_time());
  obj->reset_pesudo_node();
}

int DTRTensorPool::update_after_pesudo_compute(vm::DTREagerBlobObject* obj) {
  // split start_tensor from the chain
  auto&& pesudo_node = obj->node->pesudo_node();
  auto&& fa = find_father(pesudo_node);
  fa->set_compute_time(fa->compute_time() - obj->node->compute_time());
  fa->reduce_cnt();
  obj->reset_pesudo_node();
  return fa->cnt();
}

Maybe<void> DTRTensorPool::update_after_evict(vm::DTREagerBlobObject* obj) {
  auto* operand = obj->compute_op();
  const auto& inputs = operand->inputs();
  const auto& outputs = operand->outputs();
  for (int i = 0; i < inputs.size(); ++i) {
    if (auto tmp = inputs[i].lock()) {
      auto dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(tmp.get());
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      if (!dtr_blob_object->is_in_memory()) { merge(dtr_blob_object->node, obj->node); }
    }
  }

  for (int i = 0; i < outputs.size(); ++i) {
    if (auto tmp = outputs[i].lock()) {
      auto dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(tmp.get());
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      if (!dtr_blob_object->is_in_memory()) { merge(obj->node, dtr_blob_object->node); }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> DTRTensorPool::update_after_pesudo_evict(vm::DTREagerBlobObject* obj) {
  // include new end_tensor in the chain
  auto* operand = obj->compute_op();
  const auto& inputs = operand->inputs();
  const auto& outputs = operand->outputs();
  for (int i = 0; i < inputs.size(); ++i) {
    if (auto tmp = inputs[i].lock()) {
      auto dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(tmp.get());
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      if (!dtr_blob_object->is_in_memory()) { pesudo_merge(dtr_blob_object->node, obj->node); }
    }
  }

  for (int i = 0; i < outputs.size(); ++i) {
    if (auto tmp = outputs[i].lock()) {
      auto dtr_blob_object = dynamic_cast<vm::DTREagerBlobObject*>(tmp.get());
      CHECK_NOTNULL_OR_RETURN(dtr_blob_object);
      if (!dtr_blob_object->is_in_memory()) { pesudo_merge(obj->node, dtr_blob_object->node); }
    }
  }
  return Maybe<void>::Ok();
}

void DTRTensorPool::set_total_memory(size_t mem) { total_memory_bytes_ = mem; }

size_t DTRTensorPool::get_total_memory() { return total_memory_bytes_; }

}  // namespace one
}  // namespace oneflow
