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

#include "oneflow/core/vm/remat/util.h"

#include <algorithm>

#include "nlohmann/json.hpp"
#include "oneflow/core/common/env_var/remat.h"
#include "oneflow/core/common/env_var/vm.h"
#include "oneflow/core/eager/tensor_storage.h"
#include "oneflow/core/vm/op_call_instruction_policy.h"
#include "oneflow/core/vm/remat/env.h"
#include "oneflow/core/vm/remat/disjoint_set.h"
#include "oneflow/user/kernels/stateful_opkernel.h"

namespace oneflow {

namespace remat {

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
  if (const char* c = std::getenv("ONEFLOW_REMAT_OP_TIME_DATASET")) {
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

}  // namespace remat

namespace vm {

Pack::Pack(const OpCallInstructionPolicy& op_call_instruction_policy)
    : op_call_instruction_policy(op_call_instruction_policy) {
  const auto save_eager_blob_object_storages = [](const auto& eager_blob_objects,
                                                  auto& storage_conatiner) {
    storage_conatiner.reserve(eager_blob_objects.size());
    for (const auto& x : eager_blob_objects) {
      storage_conatiner.emplace_back(
          std::dynamic_pointer_cast<RematableTensorStorage>(x->tensor_storage()));
    }
  };
  save_eager_blob_object_storages(op_call_instruction_policy.inputs(), input_storages);
  save_eager_blob_object_storages(op_call_instruction_policy.outputs(), output_storages);
}

Maybe<void> _IncReferenceNumOfRecomputedTensor(
    const Pack& pack, int& pinned_num, std::set<const DtrOpCallInstructionPolicy*>& visited_ops) {
  VLOG(1) << "op is " << pack.op_call_instruction_policy.opkernel().op_type_name();
  for (int i = 0; i < pack.input_storages.size(); i++) {
    auto& storage = pack.input_storages[i];
    storage->Pin();
    VLOG(1) << "No." << i << " input is in memory? " << storage->is_in_memory();
    if (!storage->is_in_memory()) {
      OpCallInstructionPolicy tmp_op = storage->compute_op();
      if (!storage->is_needed_by_backward()) {
        Singleton<remat::Env>::Get()->need_eager_eviction_storages.insert(storage.get());
      }

      if (visited_ops.find(storage->dtr_compute_op().get()) == visited_ops.end()) {
        visited_ops.insert(storage->dtr_compute_op().get());
        JUST(_IncReferenceNumOfRecomputedTensor(Pack(tmp_op), pinned_num, visited_ops));
      }
    } else {
      pinned_num++;
    }
  }
  VLOG(1) << "op " << pack.op_call_instruction_policy.opkernel().op_type_name() << " end";
  return Maybe<void>::Ok();
}

Maybe<int> IncReferenceNumOfRecomputedTensor(const Pack& pack) {
  int pinned_num = 0;
  std::set<const DtrOpCallInstructionPolicy*> visited_ops;
  JUST(_IncReferenceNumOfRecomputedTensor(pack, pinned_num, visited_ops));
  return pinned_num;
}

Maybe<void> RematInputs(
    const Pack& pack, vm::Stream* vm_stream, bool first,
    const std::function<Maybe<void>(OpCallInstructionPolicy*, vm::Stream*)>& compute_fn) {
  CHECK_OR_RETURN(!ThreadLocalEnvBool<ONEFLOW_VM_MULTI_THREAD>());
  Singleton<remat::Env>::Get()->current_op_type_name =
      pack.op_call_instruction_policy.opkernel().op_type_name();
  VLOG(2) << "set current op type name to " << Singleton<remat::Env>::Get()->current_op_type_name
          << std::endl;
  if (first) { JUST(IncReferenceNumOfRecomputedTensor(pack)); }
  VLOG(1) << "compute " << pack.op_call_instruction_policy.opkernel().op_type_name() << std::endl;
  VLOG(1) << "input num " << pack.op_call_instruction_policy.inputs().size() << std::endl;

  for (int i = 0; i < pack.input_storages.size(); i++) {
    auto& storage = pack.input_storages[i];
    if (!storage->is_in_memory()) {
      VLOG(1) << "recompute No." << i << " input by " << storage->compute_op_type_name()
              << ". Storage id: " << storage->id();
      OpCallInstructionPolicy tmp_op = storage->compute_op();
      JUST(compute_fn(&tmp_op, vm_stream));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> EagerlyEvictRemattedTensors(const Pack& pack, vm::Stream* vm_stream, bool first) {
  auto& need_eager_eviction_storages = Singleton<remat::Env>::Get()->need_eager_eviction_storages;
  for (auto& storage : pack.input_storages) {
    storage->Unpin();
    if (storage->num_pinned() == 0 && need_eager_eviction_storages.count(storage.get()) > 0) {
      need_eager_eviction_storages.erase(storage.get());
      storage->Evict(true);
    }
  }
  if (first) {
    if (!need_eager_eviction_storages.empty()) {
      for (const auto& storage : need_eager_eviction_storages) {
        VLOG(1) << "not empty, storage id: " << storage->id();
      }
    }
    CHECK_OR_RETURN(need_eager_eviction_storages.empty());
  }
  return Maybe<void>::Ok();
}

Maybe<void> UpdateRematInfo(const Pack& pack, vm::Stream* vm_stream, bool first, bool recompute,
                            bool include_input, bool include_output,
                            const std::vector<bool>& storage_is_initialized) {
  if (include_output) {
    const std::unique_ptr<OpCallInstructionPolicy> compute_op = [&]() {
      auto compute_op = std::make_unique<OpCallInstructionPolicy>(pack.op_call_instruction_policy);
      for (int i = 0; i < pack.output_storages.size(); i++) {
        const auto& storage = pack.output_storages[i];
        VLOG(1) << "output " << i << " storage id: " << storage->id();
        if (storage->is_eviction_disabled()) { continue; }
        if (storage_is_initialized[i] && !recompute) {
          VLOG(1) << "storage->is_initialized(), op is " << storage->compute_op_type_name()
                  << std::endl;
          compute_op = std::make_unique<OpCallInstructionPolicy>(
              Singleton<remat::Env>::Get()->update_tensor_with_storage(
                  storage.get(), pack.op_call_instruction_policy));
        }
      }
      return compute_op;
    }();
    std::shared_ptr<DtrOpCallInstructionPolicy> dtr_compute_op =
        std::make_shared<DtrOpCallInstructionPolicy>(*compute_op);
    double compute_time = JUST(remat::GetComputeTime(*compute_op));
    for (auto& storage : pack.output_storages) {
      storage->Pin();
      if (!recompute && !storage->is_eviction_disabled()) {
        storage->set_compute_op(dtr_compute_op, compute_time);
      }
      storage->Unpin();
      storage->Access();
      remat::DisjointSet::update_after_compute(storage.get());
    }
  }
  if (include_input) {
    for (int i : pack.op_call_instruction_policy.opkernel().input_tuple_indexes4mut_ibns()) {
      pack.input_storages[i]->set_eviction_disabled(true);
    }

    for (auto& storage : pack.input_storages) { storage->Access(); }
  }

  if (recompute) { Singleton<remat::Env>::Get()->add_recomputation_num(); }
  Singleton<remat::Env>::Get()->add_time(
      JUST(remat::GetComputeTime(pack.op_call_instruction_policy)));
  VLOG(1) << "end compute " << pack.op_call_instruction_policy.opkernel().op_type_name()
          << std::endl;
  Singleton<remat::Env>::Get()->current_op_type_name = "None";
  return Maybe<void>::Ok();
}
}  // namespace vm

}  // namespace oneflow
