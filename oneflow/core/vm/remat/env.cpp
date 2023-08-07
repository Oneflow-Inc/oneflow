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
#include "oneflow/core/vm/remat/env.h"

#include "nlohmann/json.hpp"
#include "oneflow/core/eager/tensor_storage.h"
#include "oneflow/core/vm/op_call_instruction_policy.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace remat {

vm::OpCallInstructionPolicy Env::update_tensor_with_storage(
    vm::RematableTensorStorage* storage, const vm::OpCallInstructionPolicy& current_compute_op) {
  // TODO: set disjnode properly
  auto new_storage = std::make_shared<vm::RematableTensorStorage>(storage->device());
  std::unordered_map<vm::EagerBlobObject*, std::shared_ptr<vm::EagerBlobObject>> old2new;
  auto update = [&new_storage, &old2new](std::shared_ptr<vm::EagerBlobObject>& old) {
    auto it = old2new.find(old.get());
    if (it != old2new.end()) {
      old = it->second;
    } else {
      auto local_tensor_meta = old->tensor_meta();
      const auto& eager_blob_object = std::make_shared<vm::EagerBlobObject>(
          std::make_shared<MemoryCase>(old->mem_case()), local_tensor_meta, old->mut_tensor_meta(),
          local_tensor_meta->dtype(), local_tensor_meta->memory_format(), new_storage);
      eager_blob_object->set_storage_offset(old->storage_offset());
      old2new.emplace(old.get(), eager_blob_object);
      old = eager_blob_object;
    }
  };
  auto update_output = [&old2new, &new_storage](std::weak_ptr<vm::EagerBlobObject>& old) {
    auto it = old2new.find(CHECK_NOTNULL(old.lock()).get());
    if (it != old2new.end()) {
      old = it->second;
    } else {
      auto old_locked = old.lock();
      auto local_tensor_meta = old_locked->tensor_meta();
      const auto& eager_blob_object = std::make_shared<vm::EagerBlobObject>(
          std::make_shared<MemoryCase>(old_locked->mem_case()), local_tensor_meta,
          old_locked->mut_tensor_meta(), local_tensor_meta->dtype(),
          local_tensor_meta->memory_format(), new_storage);
      eager_blob_object->set_storage_offset(old_locked->storage_offset());
      old2new.emplace(old_locked.get(), eager_blob_object);
      old = eager_blob_object;
    }
  };
  for (int i = ops.size() - 1; i >= 0; i--) {
    auto& op = ops[i];
    for (int j = 0; j < op->mut_inputs().size(); j++) {
      auto& x = op->mut_inputs()[j];
      if (x == nullptr) {
        LOG(INFO) << "No." << j << " input of " << op->opkernel().op_type_name() << " is nullptr"
                  << std::endl;
        continue;
      }
      if (x->tensor_storage().get() == storage) {
        vm::EagerBlobObject* old_ptr = x.get();
        update(x);
        VLOG(1) << "update input of " << op->opkernel().op_type_name() << " from " << old_ptr
                << " (storage " << storage << ") to " << x.get() << " (storage "
                << new_storage.get() << "), op addr " << op << std::endl;
      }
    }
    for (int j = 0; j < op->mut_outputs().size(); j++) {
      auto& y = op->mut_outputs()[j];
      if (y.lock() == nullptr) {
        LOG(INFO) << "No." << j << " output of " << op->opkernel().op_type_name() << " is nullptr"
                  << std::endl;
        continue;
      }
      if (CHECK_NOTNULL(y.lock())->tensor_storage().get() == storage) {
        vm::EagerBlobObject* old_ptr = y.lock().get();
        update_output(y);
        VLOG(1) << "update output of " << op->opkernel().op_type_name() << " from " << old_ptr
                << " (storage " << storage << ") to " << y.lock().get() << " (storage "
                << new_storage.get() << "), op addr " << op << std::endl;
      }
    }
  }
  vm::OpCallInstructionPolicy new_compute_op = current_compute_op;
  // only update inputs
  for (auto& x : new_compute_op.mut_inputs()) {
    if (x->tensor_storage().get() == storage) {
      vm::EagerBlobObject* old_ptr = x.get();
      update(x);
      VLOG(1) << "update input of " << new_compute_op.opkernel().op_type_name() << " from "
              << old_ptr << " to " << x.get() << std::endl;
    }
  }
  VLOG(1) << "update_tensor_with_storage: storage " << storage->id();
  // set compute_op_ and compute_time_
  new_storage->set_compute_op(storage->dtr_compute_op(), storage->compute_time());
  // set blob_bytes_
  new_storage->set_blob_dptr(nullptr, storage->blob_bytes());
  // set is_initialized_
  new_storage->set_initialized();
  // set last_access_time_
  new_storage->Access();
  storage->clear_compute_op();
  return new_compute_op;
}

void Env::add_eviction_num(bool eager_eviction) {
  if (eager_eviction) {
    eager_eviction_num_++;
  } else {
    forced_eviction_num_++;
  }
}

Env::~Env() {
  LOG(INFO) << "forced eviction num: " << forced_eviction_num_;
  LOG(INFO) << "eager eviction num: " << eager_eviction_num_;
  LOG(INFO) << "recomputation num: " << recomputation_num_;
  LOG(INFO) << "duration: " << time_now_;

  const char* prefix = std::getenv("ONEFLOW_REMAT_SUMMARY_FILE_PREFIX");
  if (prefix != nullptr && GlobalProcessCtx::LocalRank() == 0) {
    using json = nlohmann::json;
    json cpp_summary{{"forced eviction", forced_eviction_num_},
                     {"eager eviction", eager_eviction_num_},
                     {"recomputation", recomputation_num_},
                     {"dataset time", time_now_}};

    json full_json;
    // std::fstream has strange default append semantic
    {
      std::ifstream fs(std::string(prefix) + ".json");
      if (fs.is_open()) { fs >> full_json; }
    }
    full_json.merge_patch(cpp_summary);
    {
      std::ofstream fs(std::string(prefix) + ".json");
      fs << full_json;
    }
  }
}

}  // namespace remat
}  // namespace oneflow
