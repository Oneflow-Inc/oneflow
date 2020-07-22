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
#ifndef ONEFLOW_CORE_GRAPH_SHARABLE_MEM_BLOCK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_SHARABLE_MEM_BLOCK_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/register/register_desc.pb.h"
#include "oneflow/core/graph/plan_task_graph.h"

namespace oneflow {

class SharableMemBlockEdge;

class SharableMemBlockNode final : public Node<SharableMemBlockNode, SharableMemBlockEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SharableMemBlockNode);
  SharableMemBlockNode(int64_t chain_id, const HashSet<const RegstDescProto*>& regst_descs);

  ~SharableMemBlockNode() = default;

  int64_t chain_id() const { return chain_id_; }
  const std::vector<const RegstDescProto*>& regst_descs() const { return regst_descs_; }

 private:
  const int64_t chain_id_;
  const std::vector<const RegstDescProto*> regst_descs_;
};

class SharableMemBlockEdge final : public Edge<SharableMemBlockNode, SharableMemBlockEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SharableMemBlockEdge);
  SharableMemBlockEdge() = default;
  ~SharableMemBlockEdge() = default;
};

class SharableMemBlockGraph final
    : public Graph<const SharableMemBlockNode, const SharableMemBlockEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SharableMemBlockGraph);
  SharableMemBlockGraph(const PlanTaskGraph& plan_task_gph,
                        const std::function<bool(const RegstDescProto&)>& IsSharable);
  ~SharableMemBlockGraph() = default;

  void ForEachSourceNodeGroup(
      const std::function<int64_t(const SharableMemBlockNode*)>& GroupBy,
      const std::function<void(const std::vector<const SharableMemBlockNode*>&)>& Handler) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_SHARABLE_MEM_BLOCK_GRAPH_H_
