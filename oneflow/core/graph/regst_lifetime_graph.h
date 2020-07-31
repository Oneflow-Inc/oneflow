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
#ifndef ONEFLOW_CORE_GRAPH_REGST_LIFETIME_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_REGST_LIFETIME_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/register/register_desc.pb.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

class RegstLifetimeNode;

class RegstLifetimeEdge final : public Edge<RegstLifetimeNode, RegstLifetimeEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstLifetimeEdge);
  RegstLifetimeEdge() = default;
  ~RegstLifetimeEdge() = default;
};

class RegstLifetimeNode final : public Node<RegstLifetimeNode, RegstLifetimeEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstLifetimeNode);
  RegstLifetimeNode(const RegstDescProto* regst_desc,
                    std::unique_ptr<HashSet<int64_t>>&& lifetime_actor_ids)
      : regst_desc_(regst_desc),
        lifetime_actor_ids_(std::move(lifetime_actor_ids)),
        byte_size_(RtRegstDesc(*regst_desc).TotalMainByteSize4AllRegst()) {
    CHECK_EQ(regst_desc->register_num(), 1);
  }
  ~RegstLifetimeNode() = default;

  int64_t regst_desc_id() const { return regst_desc().regst_desc_id(); }
  const RegstDescProto& regst_desc() const { return *regst_desc_; }
  const HashSet<int64_t>& lifetime_actor_ids() const { return *lifetime_actor_ids_; }
  size_t byte_size() const { return byte_size_; }

 private:
  const RegstDescProto* regst_desc_;
  std::unique_ptr<HashSet<int64_t>> lifetime_actor_ids_;
  size_t byte_size_;
};

class RegstLifetimeGraph final : public Graph<const RegstLifetimeNode, RegstLifetimeEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstLifetimeGraph);
  RegstLifetimeGraph(
      const std::vector<const RegstDescProto*>& regst_descs,
      const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>& ComputeLifetimeActorIds);
  ~RegstLifetimeGraph() = default;

  void ForEachSameColoredRegstDescs(
      const std::function<void(const std::vector<const RegstDescProto*>&)>& Handler) const;

 private:
  void InitNodes(
      const std::vector<const RegstDescProto*>& regst_descs,
      const std::function<void(const RegstDescProto*, HashSet<int64_t>*)>& ComputeLifetimeActorIds,
      std::vector<RegstLifetimeNode*>* nodes);
  void InitEdges(const std::vector<RegstLifetimeNode*>& nodes);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_REGST_LIFETIME_GRAPH_H_
