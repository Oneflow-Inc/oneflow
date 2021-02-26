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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/graph/logical_graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/op_conf_util.h"

namespace oneflow {

LogicalGraph::LogicalGraph(const Job& job) : job_(job) {
  BuildFwStruct();
  MergeEdge();
  SetNodeDataLbi();
  if (Global<ResourceDesc, ForSession>::Get()->enable_debug_mode()) { ToDotWithAutoFilePath(); }
}

template<typename LogicalNodeType>
void LogicalGraph::ForEachLogicalNode(std::function<void(LogicalNodeType*)> func) {
  std::vector<LogicalNodeType*> valid_nodes;
  ForEachNode([&](LogicalNode* logical_node) {
    auto valid_node = dynamic_cast<LogicalNodeType*>(logical_node);
    if (valid_node != nullptr) { valid_nodes.push_back(valid_node); }
  });
  for (LogicalNodeType* valid_node : valid_nodes) { func(valid_node); }
}

void LogicalGraph::BuildFwStruct() {
  HashMap<std::string, std::vector<LogicalNode*>> op_name2nodes;
  NaiveBuildFwStruct(&op_name2nodes);
}

void LogicalGraph::NaiveBuildFwStruct(
    HashMap<std::string, std::vector<LogicalNode*>>* op_name2nodes) {
  const DLNetConf& dlnet_conf = job_.net();
  const Placement& placement = job_.placement();
  HashMap<std::string, std::shared_ptr<ParallelDesc>> name2parallel_desc;
  for (const PlacementGroup& p_group : placement.placement_group()) {
    for (const std::string& op_name : p_group.op_set().op_name()) {
      CHECK(name2parallel_desc
                .emplace(op_name, std::make_shared<ParallelDesc>(p_group.parallel_conf()))
                .second);
    }
  }

  HashMap<LogicalBlobId, std::string> lbi2obn;
  HashMap<LogicalBlobId, LogicalNode*> lbi2producer;
  for (OperatorConf cur_op_conf : dlnet_conf.op()) {
    auto parallel_desc_ptr_it = name2parallel_desc.find(cur_op_conf.name());
    CHECK(parallel_desc_ptr_it != name2parallel_desc.end());
    const std::shared_ptr<ParallelDesc>& parallel_desc_ptr = parallel_desc_ptr_it->second;
    cur_op_conf.set_device_tag(CHECK_JUST(DeviceTag4DeviceType(parallel_desc_ptr->device_type())));
    std::shared_ptr<const Operator> cur_op =
        Global<OpGraph>::Get()->OpNode4OpName(cur_op_conf.name())->shared_op();
    LogicalNode* cur_node = cur_op->NewProperLogicalNode();
    AddAllocatedNode(cur_node);
    cur_node->mut_op_vec() = {cur_op};
    cur_node->mut_parallel_desc() = parallel_desc_ptr;
    {
      const auto& name2shape = job_.helper().op_name2op_time_shape();
      const auto& op_time_shape_it = name2shape.find(cur_op->op_name());
      if (op_time_shape_it != name2shape.end()) {
        const auto& op_time_shape = op_time_shape_it->second;
        if (op_time_shape.has_out_blob_time_shape()) {
          cur_node->reset_out_blob_time_shape(new Shape(op_time_shape.out_blob_time_shape()));
        }
        if (op_time_shape.has_in_blob_fastest_time_shape()) {
          cur_node->reset_in_blob_fastest_time_shape(
              new Shape(op_time_shape.in_blob_fastest_time_shape()));
        }
      }
    }
    for (const std::string& obn : cur_node->SoleOp()->output_bns()) {
      const LogicalBlobId& lbi = cur_node->SoleOp()->BnInOp2Lbi(obn);
      CHECK(lbi2producer.emplace(lbi, cur_node).second);
      CHECK(lbi2obn.emplace(lbi, obn).second);
    }
    (*op_name2nodes)[cur_op->op_name()].push_back(cur_node);
  }
  ForEachNode([&](LogicalNode* cur_node) {
    for (const std::string& ibn : cur_node->SoleOp()->input_bns()) {
      const LogicalBlobId& lbi = cur_node->SoleOp()->BnInOp2Lbi(ibn);
      LogicalNode* pred_node = lbi2producer.at(lbi);
      if (pred_node == cur_node) { continue; }
      LogicalEdge* edge = NewEdge();
      edge->mut_lbis() = {lbi};
      UpdateEdge2Ibn(edge, ibn);
      UpdateEdge2Obn(edge, lbi2obn.at(lbi));
      Connect(pred_node, edge, cur_node);
    }
  });
}

void LogicalGraph::MergeEdge() {
  ForEachNode([](LogicalNode* node) {
    HashMap<LogicalNode*, std::vector<LogicalEdge*>> dst2edges;
    for (LogicalEdge* out_edge : node->out_edges()) {
      dst2edges[out_edge->dst_node()].push_back(out_edge);
    }
    for (const auto& pair : dst2edges) {
      std::vector<LogicalBlobId>& lbi_all = pair.second.at(0)->mut_lbis();
      FOR_RANGE(size_t, i, 1, pair.second.size()) {
        std::vector<LogicalBlobId>& lbi_i = pair.second.at(i)->mut_lbis();
        lbi_all.insert(lbi_all.end(), lbi_i.begin(), lbi_i.end());
        lbi_i.clear();
        DisConnect(pair.second.at(i));  // TODO: delete its memory ?
      }
    }
  });
}

void LogicalGraph::SetNodeDataLbi() {
  ForEachNode([](LogicalNode* node) {
    for (LogicalEdge* out_edge : node->out_edges()) {
      node->SetDataLbisTo(out_edge->dst_node(), out_edge->lbis());
    }
  });
}

void LogicalGraph::ForEachNecessaryCtrlEdge(
    const std::function<void(const LogicalNode*, const LogicalNode*, int64_t)>& Handler) const {
  HashMap<std::string, const LogicalNode*> op_name2node;
  ForEachNode([&](LogicalNode* node) {
    for (const auto& op : node->op_vec()) {
      CHECK(op_name2node.emplace(op->op_name(), node).second);
    }
  });
  auto IsReachable = MakePredicatorIsReachable();
  ForEachNode([&](LogicalNode* dst) {
    for (const auto& op : dst->op_vec()) {
      for (const auto& ctrl_in_op_name : op->op_conf().ctrl_in_op_name()) {
        const LogicalNode* src = op_name2node.at(ctrl_in_op_name);
        CHECK(!IsReachable(dst, src));
        if (!IsReachable(src, dst)) {
          CHECK(src->parallel_desc()->EqualsIgnoringDeviceType(*dst->parallel_desc()));
          const Shape* src_time_shape = src->out_blob_time_shape();
          if (src_time_shape == nullptr) { src_time_shape = src->in_blob_fastest_time_shape(); }
          CHECK_NOTNULL(src_time_shape);
          const Shape* dst_time_shape = dst->in_blob_fastest_time_shape();
          if (dst_time_shape == nullptr) { dst_time_shape = dst->out_blob_time_shape(); }
          CHECK_NOTNULL(dst_time_shape);
          CHECK(src_time_shape->elem_cnt() == dst_time_shape->elem_cnt()
                || src_time_shape->Containing(*dst_time_shape));
          CHECK_EQ(src_time_shape->elem_cnt() % dst_time_shape->elem_cnt(), 0);
          int64_t regst_desc_num = src_time_shape->elem_cnt() / dst_time_shape->elem_cnt();
          Handler(src, dst, regst_desc_num);
        }
      }
    }
  });
}

void LogicalGraph::UpdateEdge2Ibn(const LogicalEdge* edge, const std::string& ibn) {
  if (!ibn.empty()) { edge2ibn_[edge] = ibn; }
}

void LogicalGraph::UpdateEdge2Obn(const LogicalEdge* edge, const std::string& obn) {
  if (!obn.empty()) { edge2obn_[edge] = obn; }
}

}  // namespace oneflow
