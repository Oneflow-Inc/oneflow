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

#include "oneflow/core/vm/remat/disjoint_set.h"

#include "oneflow/core/vm/op_call_instruction_policy.h"
#include "oneflow/core/eager/tensor_storage.h"
#include "oneflow/core/vm/remat/allocator.h"

namespace oneflow {

namespace remat {

void DisjointSet::merge(std::shared_ptr<DisjNode>& x, std::shared_ptr<DisjNode>& y) {
  auto parent_x = find_father(x);
  auto parent_y = find_father(y);
  if (parent_x.get() == parent_y.get()) { return; }

  parent_y->set_compute_time(parent_y->compute_time() + parent_x->compute_time());
  parent_x->set_parent(parent_y);
}

std::shared_ptr<DisjNode> DisjointSet::find_father(std::shared_ptr<DisjNode>& x) {
  if (x->is_root()) {
    return x;
  } else {
    auto fa = x->parent();
    auto y = find_father(fa);
    x->set_parent(y);
    return y;
  }
}

void DisjointSet::update_after_compute(vm::RematableTensorStorage* obj) {
  auto fa = find_father(obj->node);
  fa->set_compute_time(fa->compute_time() - obj->node->compute_time());
  obj->node->reset(obj->compute_time());
}

Maybe<void> DisjointSet::update_after_release(vm::RematableTensorStorage* obj) {
  CHECK_NOTNULL_OR_RETURN(obj);
  if (obj->is_eviction_disabled()) { return Maybe<void>::Ok(); }

  const auto merge_nodes = [&obj](const auto& eager_blob_objects) {
    for (int i = 0; i < eager_blob_objects.size(); ++i) {
      if (auto storage = std::dynamic_pointer_cast<vm::RematableTensorStorage>(
              eager_blob_objects[i]->tensor_storage());
          storage && !storage->is_in_memory()) {
        merge(storage->node, obj->node);
      }
    }
  };

  auto operand = obj->compute_op();
  merge_nodes(operand.inputs());
  merge_nodes(operand.outputs());

  return Maybe<void>::Ok();
}

}  // namespace remat
}  // namespace oneflow
