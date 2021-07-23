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
#ifndef ONEFLOW_CORE_GRAPH_CHAIN_ACT_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_ACT_GRAPH_H_

#include "oneflow/core/graph/graph.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/actor/act_event.pb.h"

namespace oneflow {

inline double Duration4ActEvent(const ActEvent& act_event) {
  return act_event.stop_time() - act_event.start_time();
}

class ChainActNode;

struct RegstAct {
  RegstAct(int64_t input_regst_desc_id, const ActEvent* input_producer_act_event,
           const std::list<const ActEvent*>& input_consumer_act_events)
      : regst_desc_id(input_regst_desc_id),
        producer_act_event(input_producer_act_event),
        consumer_act_events(input_consumer_act_events) {}

  int64_t regst_desc_id;
  const ActEvent* producer_act_event;
  std::list<const ActEvent*> consumer_act_events;
  std::set<const ChainActNode*> fake_producer_outs;
};

struct RegstActGroupCtx {
  RegstActGroupCtx(const std::list<const RegstAct*>& input_regst_act_group,
                   const ChainActNode* producer)
      : regst_act_group(input_regst_act_group) {
    node2duration_to_producer[producer] = 0;
  }

  const std::list<const RegstAct*>& regst_act_group;
  HashMap<const ChainActNode*, double> node2duration_to_producer;
};

class ChainActEdge final : public Edge<ChainActNode, ChainActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainActEdge);
  ChainActEdge() = delete;
  ~ChainActEdge() = default;
  ChainActEdge(double duration) : duration_(duration) {}

  // Getters
  const double duration() const { return duration_; }

 private:
  const double duration_;
};

class ChainActNode final : public Node<ChainActNode, ChainActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainActNode);
  ChainActNode() = delete;
  ~ChainActNode() = default;
  explicit ChainActNode(std::pair<int64_t, int64_t> chain_act_id_pair,
                        std::list<std::unique_ptr<ActEvent>>&& act_events);

  std::string VisualStr() const override;

  // ForEach
  void ForEachInEdge(const std::function<void(const ChainActEdge*)>& Handler) const;
  void ForEachOutEdge(const std::function<void(const ChainActEdge*)>& Handler) const;
  void ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const;
  void ForEachLastConsumedRegstAct(const std::function<void(const RegstAct*)>& Handler) const;
  void ForEachProducedRegstActGroup(
      const std::function<void(const std::list<const RegstAct*>&)>& Handler) const;

  // Getters
  int64_t act_id() const { return chain_act_id_pair_.second; }
  int64_t chain_id() const { return chain_act_id_pair_.first; }

  // Adds
  void AddProducedRegstAct(std::unique_ptr<RegstAct>&& regst_act);
  void AddLastConsumedRegstActGroup(const std::list<const RegstAct*>& regst_act_group) {
    last_consumed_regst_act_groups_.push_back(regst_act_group);
  }

 private:
  std::pair<int64_t, int64_t> chain_act_id_pair_;
  std::list<std::unique_ptr<ActEvent>> act_events_;
  std::list<std::unique_ptr<RegstAct>> produced_regst_acts_;
  std::map<std::set<const ChainActNode*>, std::list<const RegstAct*>>
      fake_outs2produced_regst_act_group_;
  std::list<std::list<const RegstAct*>> last_consumed_regst_act_groups_;
};

class ChainActSubGraph final : public Graph<const ChainActNode, const ChainActEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainActSubGraph);
  ChainActSubGraph() = delete;
  ~ChainActSubGraph() = default;
  ChainActSubGraph(const HashMap<int64_t, const TaskProto&>& task_id2task_proto,
                   std::list<std::unique_ptr<ActEvent>>&& act_events);
  const char* TypeName() const override { return "ChainActSubGraph"; }

  // Getters
  const TaskProto& GetTaskProto(int64_t actor_id) const { return task_id2task_proto_.at(actor_id); }
  bool IsActEventWithConsumer(const ActEvent* act_event) const;

  // ForEach
  void ForEachActEvent(const std::function<void(const ActEvent*)>& Handler) const;
  void ForEachRegstActConsumerPathDuration(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;

 private:
  std::function<int64_t(const ChainActNode*)> MakeGetterTopoOrderValue4Node() const;
  const ChainActNode* Node4ActEvent(const ActEvent* act_event) const;
  void InitNodes(
      std::list<std::unique_ptr<ActEvent>>&& act_events,
      HashMap<std::pair<int64_t, int64_t>, const ActEvent*>* regst_uid2producer_act_event);
  void InitEdges(
      const HashMap<std::pair<int64_t, int64_t>, const ActEvent*>& regst_uid2producer_act_event,
      HashMap<std::pair<int64_t, int64_t>, std::list<const ActEvent*>>*
          regst_uid2consumer_act_events);
  void InitNodeProducedRegstAct(
      const HashMap<std::pair<int64_t, int64_t>, const ActEvent*>& regst_uid2producer_act_event,
      const HashMap<std::pair<int64_t, int64_t>, std::list<const ActEvent*>>&
          regst_uid2consumer_act_events) const;
  void InitNodeLastConsumedRegstActGroup() const;
  void TopoForEachChainActNode(const std::function<void(const ChainActNode*)>& Handler) const;
  void CalcRegstActNodePathDuration(RegstActGroupCtx* regst_act_group_ctx,
                                    const ChainActNode* node) const;

  HashSet<const ActEvent*> act_event_with_consumer_;
  const HashMap<int64_t, const TaskProto&>& task_id2task_proto_;
  HashMap<const ActEvent*, ChainActNode*> act_event2chain_node_;
};

class ChainActGraph final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChainActGraph);
  ChainActGraph() = delete;
  ~ChainActGraph() = default;
  ChainActGraph(const Plan& plan, std::list<std::unique_ptr<ActEvent>>&& act_events);

  // Getter
  const TaskProto& GetTaskProto(int64_t actor_id) const { return task_id2task_proto_.at(actor_id); }

  // ForEach
  void ForEachRegstDescConsumerPathMeanDuration(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;
  void ForEachRegstDescConsumerPathIIScale(
      const std::function<void(int64_t, int64_t, double)>& Handler) const;

  double CalcBaseII() const;

 private:
  void InitTaskId2TaskProto();
  void ForEachChainActSubGraph(const std::function<void(const ChainActSubGraph*)>& Handler) const;
  void GroupActEventByActId(
      std::list<std::unique_ptr<ActEvent>>&& act_events,
      HashMap<int64_t, std::list<std::unique_ptr<ActEvent>>>* act_id2act_event_group) const;
  void MultiThreadBuildChainActSubGraph(
      HashMap<int64_t, std::list<std::unique_ptr<ActEvent>>>* act_id2act_event_group);

  const Plan* plan_;
  HashMap<int64_t, const TaskProto&> task_id2task_proto_;
  std::list<std::unique_ptr<ChainActSubGraph>> sub_graphs_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_ACT_GRAPH_H_
