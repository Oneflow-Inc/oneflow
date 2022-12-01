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

#include "oneflow/core/vm/dtr_disjoint_set.h"

#include "oneflow/core/vm/op_call_instruction_policy.h"
#include "oneflow/core/eager/tensor_storage.h"
#include "oneflow/core/vm/dtr_ep_allocator.h"

namespace oneflow {

namespace dtr {

void DisjointSet::merge(std::shared_ptr<DisjNode>& x, std::shared_ptr<DisjNode>& y) {
  auto&& parent_x = find_father(x);
  auto&& parent_y = find_father(y);
  if (parent_x.get() == parent_y.get()) { return; }

  parent_y->set_compute_time(parent_y->compute_time() + parent_x->compute_time());
  parent_x->set_parent(parent_y);
}

std::shared_ptr<DisjNode> DisjointSet::find_father(std::shared_ptr<DisjNode>& x) {
  if (x->is_root()) {
    return x;
  } else {
    auto fa = x->parent();
    auto&& y = find_father(fa);
    x->set_parent(y);
    return y;
  }
}

void DisjointSet::update_after_compute(vm::TensorStorage* obj) {
  auto&& fa = find_father(obj->node);
  fa->set_compute_time(fa->compute_time() - obj->node->compute_time());
  obj->node->reset(obj->compute_time());
}

Maybe<void> DisjointSet::update_after_evict(vm::TensorStorage* obj) {
  auto operand = obj->compute_op();
  const auto& inputs = operand.inputs();
  const auto& outputs = operand.outputs();
  for (int i = 0; i < inputs.size(); ++i) {
    auto storage = inputs[i]->tensor_storage();
    if (!storage->is_in_memory()) { merge(storage->node, obj->node); }
  }

  for (int i = 0; i < outputs.size(); ++i) {
    auto storage = outputs[i]->tensor_storage();
    if (!storage->is_in_memory()) { merge(obj->node, storage->node); }
  }
  return Maybe<void>::Ok();
}

}  // namespace dtr
}  // namespace oneflow
