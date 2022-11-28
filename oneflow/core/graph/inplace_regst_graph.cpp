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
#include "oneflow/core/graph/inplace_regst_graph.h"

namespace oneflow {

namespace {

std::function<const RegstDescProto*(int64_t)> MakeGetterRegstDesc4RegstDescId(
    const HashSet<const RegstDescProto*>& regst_descs) {
  auto regst_desc_id2regst_desc = std::make_shared<HashMap<int64_t, const RegstDescProto*>>();
  for (const auto* regst_desc : regst_descs) {
    CHECK(regst_desc_id2regst_desc->emplace(regst_desc->regst_desc_id(), regst_desc).second);
  }
  return [regst_desc_id2regst_desc](int64_t regst_desc_id) -> const RegstDescProto* {
    auto it = regst_desc_id2regst_desc->find(regst_desc_id);
    return it == regst_desc_id2regst_desc->end() ? nullptr : it->second;
  };
}

}  // namespace

InplaceRegstGraph::InplaceRegstGraph(const HashSet<const RegstDescProto*>& regst_descs) {
  auto RegstDesc4RegstDescId = MakeGetterRegstDesc4RegstDescId(regst_descs);
  auto FindOrCreate = MakeMutFindOrCreateNode();
  for (const RegstDescProto* regst_desc : regst_descs) {
    if (regst_desc->has_hint_inplace_consumed_regst_desc_id()) {
      const RegstDescProto* in_regst_desc =
          RegstDesc4RegstDescId(regst_desc->hint_inplace_consumed_regst_desc_id());
      if (in_regst_desc != nullptr) {
        auto* edge = new InplaceRegstEdge();
        AddAllocatedEdge(edge);
        Connect<InplaceRegstNode, InplaceRegstEdge>(FindOrCreate(in_regst_desc), edge,
                                                    FindOrCreate(regst_desc));
      }
    }
  }
}

std::function<InplaceRegstNode*(const RegstDescProto*)>
InplaceRegstGraph::MakeMutFindOrCreateNode() {
  auto regst_desc2node = std::make_shared<HashMap<const RegstDescProto*, InplaceRegstNode*>>();
  return [regst_desc2node, this](const RegstDescProto* regst_desc) -> InplaceRegstNode* {
    auto it = regst_desc2node->find(regst_desc);
    if (it == regst_desc2node->end()) {
      InplaceRegstNode* node = new InplaceRegstNode(regst_desc);
      AddAllocatedNode(node);
      it = regst_desc2node->emplace(regst_desc, node).first;
    }
    return it->second;
  };
}

}  // namespace oneflow
