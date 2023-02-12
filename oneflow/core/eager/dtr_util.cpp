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

#include "oneflow/core/eager/dtr_util.h"

#include <algorithm>

#include "nlohmann/json.hpp"
#include "oneflow/core/common/env_var/dtr.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/vm/dtr_ep_allocator.h"
#include "oneflow/core/vm/ep_backend_allocator.h"
#include "oneflow/core/vm/op_call_instruction_policy.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/user/kernels/stateful_opkernel.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/eager/tensor_storage.h"

namespace oneflow {

namespace dtr {

size_t memory_threshold() { return EnvInteger<ONEFLOW_DTR_BUDGET_MB>() * 1024 * 1024; }

bool is_enabled_and_debug() { return is_enabled() && debug_level() > 0; }

int debug_level() {
  if (!is_enabled()) { return 0; }
  return EnvInteger<ONEFLOW_DTR_DEBUG_LEVEL>();
}

double append_memory_frag_info_and_get(size_t free_mem, size_t threshold) {
  static size_t num = 0;
  // maintain a summation of memory frag rate
  static double memory_frag_rate_sum = 0;
  if (threshold > 0) {
    memory_frag_rate_sum += (1. * free_mem / threshold);
    num++;
  }
  return memory_frag_rate_sum / num;
}

vm::DtrEpAllocator* AllocatorManager::CreateOrGetAllocator(DeviceType device_type,
                                                           size_t device_index) {
  auto key = std::make_pair(device_type, device_index);
  auto it = allocators_.find(key);
  if (it == allocators_.end()) {
    auto ep_device =
        Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(device_type, device_index);
    auto ep_backend_allocator =
        std::make_unique<vm::EpBackendAllocator>(ep_device, ep::AllocationOptions{});
    auto allocator = std::make_unique<vm::DtrEpAllocator>(ep::kMaxAlignmentRequirement,
                                                          std::move(ep_backend_allocator));
    allocators_.emplace(key, std::move(allocator));
    return allocators_.at(key).get();
  } else {
    return it->second.get();
  }
}

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
          local_tensor_meta->dtype(), new_storage);
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
          old_locked->mut_tensor_meta(), local_tensor_meta->dtype(), new_storage);
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

  const char* prefix = std::getenv("ONEFLOW_DTR_SUMMARY_FILE_PREFIX");
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

namespace {

std::string SortKey(const std::string& key) {
  const auto shape_finish_at = key.rfind(")");
  if (shape_finish_at == std::string::npos || shape_finish_at + 2 == key.size()) { return key; }
  const auto name_and_shape = key.substr(0, shape_finish_at + 1);
  auto attrs = key.substr(shape_finish_at + 2);
  if (attrs.substr(attrs.size() - 2) == ", ") { attrs = attrs.substr(0, attrs.size() - 2); }

  const auto need_find_next = [](const std::string& s, size_t index) {
    const size_t final_pos = index + 2;
    if (final_pos >= s.size()) { return false; }
    if (s.at(index + 1) != ' ') { return true; }
    if (!(s.at(final_pos) >= 'a' && s.at(final_pos) <= 'z')) { return true; }
    return false;
  };

  const auto split = [&need_find_next](const std::string& s, std::vector<std::string>& tokens,
                                       const std::string& delimiters) {
    std::string::size_type lastPos = s.find_first_not_of(delimiters, 0);
    std::string::size_type pos = s.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos && need_find_next(s, pos)) {
      pos = s.find_first_of(delimiters, pos + 1);
    }
    while (std::string::npos != pos || std::string::npos != lastPos) {
      tokens.push_back(s.substr(lastPos, pos - lastPos));
      lastPos = s.find_first_not_of(delimiters, pos);
      pos = s.find_first_of(delimiters, lastPos);
      while (std::string::npos != pos && need_find_next(s, pos)) {
        pos = s.find_first_of(delimiters, pos + 1);
      }
    }
  };
  std::vector<std::string> attrs_splited;
  split(attrs, attrs_splited, ", ");
  std::sort(attrs_splited.begin(), attrs_splited.end());
  return fmt::format("{} {}, ", name_and_shape, fmt::join(attrs_splited, ", "));
}

using json = nlohmann::json;

json LoadTimeDataset() {
  json j;
  if (const char* c = std::getenv("ONEFLOW_DTR_OP_TIME_DATASET")) {
    std::ifstream i(c);
    i >> j;
    i.close();
  }
  json new_j;

  for (json::iterator iter = j.begin(); iter != j.end(); ++iter) {
    new_j[SortKey(iter.key())] = iter.value();
  }
  return new_j;
}

Maybe<double> GetDatasetComputeTime(const json& j, const vm::OpCallInstructionPolicy& operand) {
  const std::vector<std::string> zero_time_list{
      "empty", "identity", "constant", "copy", "zero_like", "expand_dims", "flatten", "reduce_sum",
      "reshape", "reshape_like", "squeeze", "transpose", "nll", "nll_grad", "uniform",
      "uniform_int", "fill_", "slice_update", "normal",
      // ddp
      "eager_ccl_broadcast", "eager_ccl_all_reduce", "eager_nccl_touch", "scalar_mul",

      // "adaptive_avg_pool2d",
      // "adaptive_avg_pool2d_grad"
  };
  for (const auto& x : zero_time_list) {
    if (operand.opkernel().op_type_name() == x) { return 0; }
  }

  const std::string op_type_str = operand.opkernel().op_type_name();
  const std::string input_shape_str = [&]() {
    std::stringstream ss;
    for (size_t i = 0; i < operand.inputs().size(); i++) {
      ss << operand.inputs().at(i)->shape();
      if (i != operand.inputs().size() - 1) { ss << ", "; }
    }
    return ss.str();
  }();
  const std::string attr_str = operand.composed_attrs().ToString();
  std::string key = op_type_str + " " + input_shape_str + " " + attr_str;
  key = SortKey(key);
  CHECK_OR_RETURN(j.contains(key)) << "key " << key << " not found";
  CHECK_OR_RETURN(j[key].is_number_float()) << "key " << key << " is not float, but " << j[key];
  return j[key].get<double>();
}

static Maybe<double> GetEstimatedComputeTime(const vm::OpCallInstructionPolicy& operand) {
  const auto& inputs = operand.inputs();
  const auto& outputs = operand.outputs();
  size_t estimated_compute_time = 0;
  for (const auto& input : inputs) {
    estimated_compute_time += input->tensor_storage()->blob_bytes();
  }
  for (const auto& output : outputs) {
    estimated_compute_time += output->tensor_storage()->blob_bytes();
  }
  return estimated_compute_time;
}
}  // namespace

Maybe<double> GetComputeTime(const vm::OpCallInstructionPolicy& operand) {
  const static json time_dataset = LoadTimeDataset();
  if (!time_dataset.empty()) { return GetDatasetComputeTime(time_dataset, operand); }
  return GetEstimatedComputeTime(operand);
}

}  // namespace dtr

}  // namespace oneflow
