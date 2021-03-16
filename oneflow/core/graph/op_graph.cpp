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
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/mirrored_sig_infer_hint.h"
#include "oneflow/core/framework/device_registry_manager.h"

namespace oneflow {

std::string OpEdge::VisualStr() const {
  std::string str;
  int32_t idx = 0;
  for (const LogicalBlobId& lbi : *lbis_) {
    if (idx++ > 0) { str += "\\n"; }
    str += lbi.blob_name() + ":";
    str += src_node()->LogicalBlobDesc4Lbi(lbi).shape().ToString();
  }
  return str;
}

void OpEdge::InitDistributeHierarchyInfo() { InitIsStrict121(); }

void OpEdge::InitIsStrict121() { is_strict_121_ = CalcIsStrict121Connected(); }

bool OpEdge::CalcIsStrict121Connected() const {
  OpNode* src = src_node();
  OpNode* dst = dst_node();
  if (!src->parallel_desc().Equals(dst->parallel_desc())) { return false; }
  if (!src->IsTimeShapeIdentity()) { return false; }
  if (!dst->IsTimeShapeIdentity()) { return false; }
  if (*CHECK_JUST(src->op().GetOpTimeShape()) != *CHECK_JUST(dst->op().GetOpTimeShape())) {
    return false;
  }
  for (const LogicalBlobId& lbi : lbis()) {
    const ParallelDistribution& obn_distribution =
        src->ParallelDistribution4BnInOp(lbi2obn().at(lbi));
    for (const std::string& ibn : lbi2ibns().at(lbi)) {
      const ParallelDistribution& ibn_distribution = dst->ParallelDistribution4BnInOp(ibn);
      if (obn_distribution != ibn_distribution) { return false; }
    }
  }
  return true;
}

const SbpParallel& OpNode::SbpParallel4BnInOp(const std::string& bn_in_op) const {
  return *CHECK_JUST(op().SbpParallel4BnInOp(bn_in_op));
}

const SbpParallel& OpNode::SbpParallel4Lbi(const LogicalBlobId& lbi) const {
  auto it = lbi2parallel_distribution_.find(lbi);
  CHECK(it != lbi2parallel_distribution_.end());
  CHECK_EQ(it->second.sbp_parallel_size(), 1);
  return it->second.sbp_parallel(0);
}

const ParallelDistribution& OpNode::ParallelDistribution4BnInOp(const std::string& bn_in_op) const {
  return *CHECK_JUST(op().ParallelDistribution4BnInOp(bn_in_op));
}

const ParallelDistribution& OpNode::ParallelDistribution4Lbi(const LogicalBlobId& lbi) const {
  auto it = lbi2parallel_distribution_.find(lbi);
  CHECK(it != lbi2parallel_distribution_.end());
  return it->second;
}

OpNode::OpNode(const std::shared_ptr<const ParallelDesc>& parallel_desc,
               const OperatorConf& op_conf)
    : parallel_desc_(parallel_desc),
      op_(ConstructOp(op_conf, parallel_desc->device_type())),
      ibns_(op_->input_bns().begin(), op_->input_bns().end()) {
  op_->FillOpParallelDesc(parallel_desc);
}

std::string OpNode::VisualStr() const {
  std::string str = op().op_name();
  {
    for (int64_t machine_id : parallel_desc().sorted_machine_ids()) {
      const std::string dev_type = *CHECK_JUST(DeviceTag4DeviceType(parallel_desc().device_type()));

      std::string parallel_desc_str = std::to_string(machine_id) + ":" + dev_type + ":";
      const auto& dev_phy_ids = parallel_desc().sorted_dev_phy_ids(machine_id);
      parallel_desc_str += std::to_string(dev_phy_ids.front());
      if (dev_phy_ids.back() > dev_phy_ids.front()) {
        parallel_desc_str += "-" + std::to_string(dev_phy_ids.back());
      }
      str += "\\n" + parallel_desc_str;
    }
  }
  auto GetTimeShapeStr = [&](const Shape& shape, const std::string& prefix) {
    std::string time_shape_str = prefix + ":";
    time_shape_str += shape.ToString();
    return time_shape_str;
  };
  if (in_edges().empty() == false) {
    str +=
        "\\n"
        + GetTimeShapeStr(*CHECK_JUST(op().GetInputBlobFastestTimeShape()), "in_blob_time_shape");
  }
  str += "\\n" + GetTimeShapeStr(*CHECK_JUST(op().GetOpTimeShape()), "op_time_shape");
  return str;
}

const BlobDesc& OpNode::LogicalBlobDesc4Lbi(const LogicalBlobId& lbi) const {
  const OpNode& producer = ProducerOpNode4Lbi(lbi);
  const int32_t index = CHECK_JUST(producer.op().GetOutputIndex(lbi));
  const BlobDesc* blob_desc = CHECK_JUST(producer.op().GetLogicalBlobDescPtr4OutputIndex(index));
  return *blob_desc;
}

const OpNode& OpNode::SrcNode4Ibn(const std::string& bn_in_op) const {
  return *MutSrcNode4Ibn(bn_in_op);
}

OpNode* OpNode::MutSrcNode4Ibn(const std::string& bn_in_op) const {
  const LogicalBlobId& lbi = op().BnInOp2Lbi(bn_in_op);
  CHECK(ibns_.find(bn_in_op) != ibns_.end());
  return MutSrcNode4InputLbi(lbi);
}

const OpNode& OpNode::ProducerOpNode4Lbi(const LogicalBlobId& lbi) const {
  const OpNode* producer = MutSrcNode4InputLbi(lbi);
  if (producer == nullptr) { producer = this; }
  return *producer;
}

OpNode* OpNode::MutSrcNode4InputLbi(const LogicalBlobId& lbi) const {
  auto it = lbi2source_node_.find(lbi);
  if (it == lbi2source_node_.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

bool OpNode::IsTimeShapeIdentity() const {
  std::shared_ptr<const Shape> in_shape = CHECK_JUST(op().GetInputBlobFastestTimeShape());
  if (!in_shape) { return true; }
  std::shared_ptr<const Shape> op_shape = CHECK_JUST(op().GetOpTimeShape());
  return *in_shape == *op_shape;
}

const ParallelDesc& OpNode::BlobParallelDesc4Obn(const std::string& obn) const {
  return obn2blob_parallel_desc_.at(obn);
}

void OpNode::InferBlobParallelDesc() {
  for (const auto& bn : op().output_bns()) {
    obn2blob_parallel_desc_.emplace(bn, *CHECK_JUST(op().GetParallelDesc4BnInOp(bn)));
  }
}

void OpNode::InitLbi2SourceNode() {
  for (OpEdge* edge : in_edges()) {
    for (const LogicalBlobId& lbi : edge->lbis()) {
      CHECK(lbi2source_node_.emplace(lbi, edge->src_node()).second);
    }
  }
}

void OpNode::InitLbi2ParallelDistribution() {
  const auto Update = [&](const PbRpf<std::string>& bns) {
    for (const auto& bn : bns) {
      const LogicalBlobId& lbi = op().BnInOp2Lbi(bn);
      const ParallelDistribution& parallel_distribution = ParallelDistribution4BnInOp(bn);
      auto it = lbi2parallel_distribution_.find(lbi);
      if (it == lbi2parallel_distribution_.end()) {
        lbi2parallel_distribution_[lbi] = parallel_distribution;
      } else {
        CHECK(it->second == parallel_distribution);
      }
    }
  };
  Update(op().input_bns());
  Update(op().output_bns());
}

Maybe<OpGraph> OpGraph::New(const Job& job) {
  const auto& op_graph = std::make_shared<OpGraph>();
  JUST(op_graph->Init(job));
  return op_graph;
}

Maybe<void> OpGraph::Init(const Job& job) {
  InitNodes(job);
  ForEachNode([&](OpNode* node) {
    CHECK(op_name2op_node_.emplace(node->op().op_name(), node).second)
        << "op_name: " << node->op().op_name();
  });
  InitEdges();
  InitProducerOpName2CtrlConsumerOpNames(job);
  CheckIsDAG();
  ForEachNode([](OpNode* node) { node->InitLbi2SourceNode(); });
  InferBlobLastUsed();
  InferTimeShape();
  JUST(InferLogicalBlobDesc(job));
  ForEachEdge([](OpEdge* edge) { edge->InitDistributeHierarchyInfo(); });
  return Maybe<void>::Ok();
}

void OpGraph::CheckIsDAG() const {
  CHECK(!FindFirstNontrivialSCC());
  auto ForEachIn = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    ForEachDataAndCtrlInNode(node, Handler);
  };
  auto ForEachOut = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    ForEachDataAndCtrlOutNode(node, Handler);
  };
  CHECK(!FindFirstNontrivialSCC(ForEachIn, ForEachOut));
}

namespace {

std::function<std::shared_ptr<const ParallelDesc>(const std::string&)>
MakeGetterParallelDesc4OpName(const Placement& placement) {
  auto op_name2parallel_desc =
      std::make_shared<HashMap<std::string, std::shared_ptr<const ParallelDesc>>>();
  for (const auto& placement_group : placement.placement_group()) {
    for (const std::string& op_name : placement_group.op_set().op_name()) {
      const ParallelConf* parallel_conf = &placement_group.parallel_conf();
      CHECK(op_name2parallel_desc
                ->emplace(op_name, std::make_shared<const ParallelDesc>(*parallel_conf))
                .second)
          << "op_name: " << op_name;
    }
  }
  return [op_name2parallel_desc](const std::string& op_name) {
    return op_name2parallel_desc->at(op_name);
  };
}

}  // namespace

void OpGraph::InitNodes(const Job& job) {
  auto ParallelDesc4OpName = MakeGetterParallelDesc4OpName(job.placement());
  for (const auto& op_conf : job.net().op()) {
    op_names_.push_back(op_conf.name());
    OpNode* node = new OpNode(ParallelDesc4OpName(op_conf.name()), op_conf);
    AddAllocatedNode(node);
  }
}

void OpGraph::InitEdges() {
  HashMap<LogicalBlobId, OpNode*> lbi2producer;
  HashMap<std::string, std::shared_ptr<HashMap<LogicalBlobId, std::string>>>
      producer_op_name2lbi2obn;
  ForEachNode([&](OpNode* op_node) {
    for (const auto& obn : op_node->op().output_bns()) {
      const auto& lbi = op_node->op().BnInOp2Lbi(obn);
      CHECK(lbi2producer.emplace(lbi, op_node).second);
      auto& lbi2obn = producer_op_name2lbi2obn[op_node->op().op_name()];
      if (!lbi2obn) { lbi2obn.reset(new HashMap<LogicalBlobId, std::string>()); }
      CHECK(lbi2obn->emplace(lbi, obn).second);
    }
  });
  ForEachNode([&](OpNode* op_node) {
    HashMap<std::string, HashSet<LogicalBlobId>> producer_op_name2lbis;
    std::shared_ptr<HashMap<LogicalBlobId, std::vector<std::string>>> consumer_lbi2ibns(
        new HashMap<LogicalBlobId, std::vector<std::string>>);
    op_node->input_index2producer_and_output_index_.reserve(op_node->op().input_bns().size());
    for (const auto& ibn : op_node->op().input_bns()) {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
      producer_op_name2lbis[lbi.op_name()].insert(lbi);
      (*consumer_lbi2ibns)[lbi].push_back(ibn);
      auto producer_it = lbi2producer.find(lbi);
      CHECK(producer_it != lbi2producer.end()) << "producer not found: " << GenLogicalBlobName(lbi);
      const int32_t output_index = CHECK_JUST(producer_it->second->op().GetOutputIndex(lbi));
      op_node->input_index2producer_and_output_index_.emplace_back(producer_it->second,
                                                                   output_index);
    }
    for (const auto& pair : producer_op_name2lbis) {
      std::shared_ptr<std::vector<LogicalBlobId>> lbis(
          new std::vector<LogicalBlobId>({pair.second.begin(), pair.second.end()}));
      const auto it = producer_op_name2lbi2obn.find(pair.first);
      CHECK(it != producer_op_name2lbi2obn.end()) << "producer_op_name: " << pair.first;
      const auto& lbi2obn = it->second;
      auto producer_it = lbi2producer.find(lbis->front());
      CHECK(producer_it != lbi2producer.end())
          << "producer not found: " << GenLogicalBlobName(lbis->front());
      Connect(producer_it->second, NewEdge(lbis, lbi2obn, consumer_lbi2ibns), op_node);
    }
  });
}

void OpGraph::InitProducerOpName2CtrlConsumerOpNames(const Job& job) {
  for (const auto& op_conf : job.net().op()) {
    for (const auto& ctrl_in_op_name : op_conf.ctrl_in_op_name()) {
      auto* consumer_op_names = &producer_op_name2ctrl_consumer_op_names_[ctrl_in_op_name];
      CHECK(consumer_op_names->emplace(op_conf.name()).second);
    }
  }
}

void OpGraph::InferBlobLastUsed() const {
  HashSet<LogicalBlobId> visisted_lbi;
  for (auto iter = op_names_.rbegin(); iter != op_names_.rend(); iter++) {
    Operator* op = op_name2op_node_.at(*iter)->mut_op();
    auto* map = op->mut_blob_last_used_signature()->mutable_bn_in_op2blob_last_used();
    const auto InferLastUsed = [&](const std::string& bn_in_op) {
      (*map)[bn_in_op] = visisted_lbi.insert(op->BnInOp2Lbi(bn_in_op)).second;
    };
    for (const auto& obn : op->output_bns()) { InferLastUsed(obn); }
    for (const auto& ibn : op->input_bns()) { InferLastUsed(ibn); }
  }
}

void OpGraph::InferTimeShape() const {
  TopoForEachNode([&](OpNode* op_node) {
    auto GetInputBlobTimeShape = [&](const std::string& bn_in_op) {
      return op_node->MutSrcNode4Ibn(bn_in_op)->op().GetOpTimeShape();
    };
    CHECK_JUST(op_node->mut_op()->FillInputBlobTimeShape(GetInputBlobTimeShape));
    CHECK_JUST(op_node->mut_op()->InferOpTimeShapeIf());
  });
}

void OpGraph::InferOpNodeParallelDistributionSignature(
    OpNode* op_node, const ParallelDistributionSignature& parallel_distribution_sig_conf) const {
  HashMap<std::string, ParallelDistributionInferHint> ibn2parallel_distribution_infer_hint;
  for (const std::string& ibn : op_node->op().input_bns()) {
    const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
    OpNode* producer = op_node->MutSrcNode4Ibn(ibn);
    const ParallelDesc* parallel_desc = &producer->parallel_desc();
    const BlobDesc* logical_blob_desc = &producer->LogicalBlobDesc4Lbi(lbi);
    const ParallelDistribution* parallel_distribution = &producer->ParallelDistribution4Lbi(lbi);
    CHECK_EQ(parallel_distribution->sbp_parallel_size(), 1);
    ibn2parallel_distribution_infer_hint.emplace(
        ibn,
        ParallelDistributionInferHint(parallel_desc, logical_blob_desc, parallel_distribution));
  }
  const auto ParallelDistributionInferHint4Ibn =
      [&](const std::string& bn) -> Maybe<const ParallelDistributionInferHint*> {
    auto it = ibn2parallel_distribution_infer_hint.find(bn);
    CHECK_OR_RETURN(it != ibn2parallel_distribution_infer_hint.end());
    return Maybe<const ParallelDistributionInferHint*>(&it->second);
  };
  CHECK_JUST(op_node->mut_op()->InferParallelDistributionSignatureIf(
      parallel_distribution_sig_conf, op_node->parallel_desc(), ParallelDistributionInferHint4Ibn));
  op_node->InitLbi2ParallelDistribution();
}

Maybe<void> OpGraph::InferOpNodeMirroredSignature(OpNode* op_node, bool is_mirrored_conf) const {
  HashMap<std::string, MirroredSigInferHint> ibn2mirrored_sig_infer_hint;
  for (const std::string& ibn : op_node->op().input_bns()) {
    const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
    const auto* producer = op_node->MutSrcNode4Ibn(ibn);
    const ParallelDesc* parallel_desc = &producer->parallel_desc();
    const auto& producer_obn = *JUST(producer->op().obn4lbi(lbi));
    const auto& opt_mirrored_parallel =
        *JUST(producer->op().OptMirroredParallel4BnInOp(producer_obn));
    MirroredSigInferHint infer_ctx(parallel_desc, opt_mirrored_parallel.has_mirrored_parallel());
    ibn2mirrored_sig_infer_hint.emplace(ibn, infer_ctx);
  }
  const auto& MirroredSigInferHint4Ibn =
      [&](const std::string& ibn) -> Maybe<const MirroredSigInferHint*> {
    const auto& iter = ibn2mirrored_sig_infer_hint.find(ibn);
    CHECK_OR_RETURN(iter != ibn2mirrored_sig_infer_hint.end())
        << "input blob not found. ibn: " << ibn;
    return &iter->second;
  };
  JUST(op_node->mut_op()->InferMirroredSignatureIf(MirroredSigInferHint4Ibn, is_mirrored_conf,
                                                   op_node->parallel_desc()));
  return Maybe<void>::Ok();
}

const OpNode* OpGraph::OpNode4OpName(const std::string& op_name) const {
  const auto& op_node_it = op_name2op_node_.find(op_name);
  if (op_node_it == op_name2op_node_.end()) { return nullptr; }
  return op_node_it->second;
}

Maybe<void> OpGraph::InferLogicalBlobDesc(const Job& job) const {
  JobParallelViewConf job_parallel_view_conf(job.job_parallel_view_conf());
  JUST(TopoForEachNodeWithErrorCaptured([&](OpNode* op_node) -> Maybe<void> {
    auto LogicalBlobDesc4InputIndex = [&](int32_t index) -> Maybe<const BlobDesc> {
      CHECK_LT_OR_RETURN(index, op_node->input_index2producer_and_output_index_.size());
      const auto& producer_info = op_node->input_index2producer_and_output_index_.at(index);
      return producer_info.first->op().GetLogicalBlobDesc4OutputIndex(producer_info.second);
    };
    JUST(op_node->mut_op()->FillLogicalInBlobDesc(LogicalBlobDesc4InputIndex));
    // Infer ParallelSignature
    JUST(op_node->mut_op()->InferParallelSignatureIf());
    // Infer mirrored_signature
    bool is_mirrored_conf = false;
    {
      const auto& op_name2is_mirrored = job_parallel_view_conf.op_name2is_mirrored_parallel_view();
      const auto& iter = op_name2is_mirrored.find(op_node->op().op_name());
      if (iter != op_name2is_mirrored.end()) { is_mirrored_conf = iter->second; }
    }
    JUST(InferOpNodeMirroredSignature(op_node, is_mirrored_conf));
    ParallelDistributionSignature parallel_distribution_sig_conf;
    {
      const auto& op_name2parallel_distribution_sig_conf =
          job_parallel_view_conf.op_name2parallel_distribution_signature_conf();
      const auto& iter = op_name2parallel_distribution_sig_conf.find(op_node->op().op_name());
      if (iter != op_name2parallel_distribution_sig_conf.end()) {
        parallel_distribution_sig_conf = iter->second;
        if (op_node->parallel_desc().hierarchy()->NumAxes() == 1) {
          const auto& op_name2sbp_sig_conf = job_parallel_view_conf.op_name2sbp_signature_conf();
          const auto& op_name2sbp_sig_conf_it = op_name2sbp_sig_conf.find(op_node->op().op_name());
          CHECK(op_name2sbp_sig_conf_it != op_name2sbp_sig_conf.end());
          CheckSbpSignatureAndParallelDistributionEquals(op_name2sbp_sig_conf_it->second,
                                                         iter->second);
        } else {
          UNIMPLEMENTED();
        }
      }
    }
    InferOpNodeParallelDistributionSignature(op_node, parallel_distribution_sig_conf);
    op_node->InferBlobParallelDesc();
    JUST(op_node->mut_op()->InferLogicalOutBlobDescsIf());
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

int64_t OpGraph::GetParallelNum(const std::string& op_name) const {
  return op_name2op_node_.at(op_name)->parallel_desc().parallel_num();
}

const SbpParallel& OpGraph::GetSbpParallel(const std::string& op_name,
                                           const LogicalBlobId& lbi) const {
  return op_name2op_node_.at(GetOpNameKey(op_name, lbi))
      ->SbpParallel4Lbi(GetLogicalBlobIdKey(op_name, lbi));
}

const ParallelDistribution& OpGraph::GetParallelDistribution(const std::string& op_name,
                                                             const LogicalBlobId& lbi) const {
  return op_name2op_node_.at(GetOpNameKey(op_name, lbi))
      ->ParallelDistribution4Lbi(GetLogicalBlobIdKey(op_name, lbi));
}

DataType OpGraph::GetBlobDataType(const LogicalBlobId& lbi) const {
  return op_name2op_node_.at(lbi.op_name())
      ->LogicalBlobDesc4Lbi(GetLogicalBlobIdKey(lbi.op_name(), lbi))
      .data_type();
}

const BlobDesc& OpGraph::GetLogicalBlobDesc(const LogicalBlobId& lbi) const {
  return op_name2op_node_.at(lbi.op_name())
      ->LogicalBlobDesc4Lbi(GetLogicalBlobIdKey(lbi.op_name(), lbi));
}

std::string OpGraph::GetOpNameKey(const std::string& op_name, const LogicalBlobId& lbi) const {
  CHECK(!lbi.has_is_packed_id());
  if (op_name2op_node_.find(op_name) != op_name2op_node_.end()) {
    return op_name;
  } else {
    UNIMPLEMENTED();
  }
}

LogicalBlobId OpGraph::GetLogicalBlobIdKey(const std::string& op_name,
                                           const LogicalBlobId& lbi) const {
  CHECK(!lbi.has_is_packed_id());
  if (op_name2op_node_.find(op_name) != op_name2op_node_.end()) {
    return lbi;
  } else {
    UNIMPLEMENTED();
  }
}

void OpGraph::ForEachDataAndCtrlInNode(OpNode* node,
                                       const std::function<void(OpNode*)>& Handler) const {
  node->ForEachNodeOnInEdge(Handler);
  for (const auto& ctrl_in_op_name : node->op().op_conf().ctrl_in_op_name()) {
    Handler(op_name2op_node_.at(ctrl_in_op_name));
  }
}

void OpGraph::ForEachDataAndCtrlOutNode(OpNode* node,
                                        const std::function<void(OpNode*)>& Handler) const {
  node->ForEachNodeOnOutEdge(Handler);
  const auto& op_name_it = producer_op_name2ctrl_consumer_op_names_.find(node->op().op_name());
  if (op_name_it == producer_op_name2ctrl_consumer_op_names_.end()) { return; }
  for (const std::string& ctrl_consumer_op_name : op_name_it->second) {
    Handler(op_name2op_node_.at(ctrl_consumer_op_name));
  }
}

std::function<bool(const std::string&, const std::string&)>
OpGraph::MakePredicatorIsOpNameDataOrCtrlReachable() const {
  auto IsDataOrCtrlReachable = MakePredicatorIsDataOrCtrlReachable();
  return [IsDataOrCtrlReachable, this](const std::string& lhs, const std::string& rhs) {
    const auto& src_node_it = op_name2op_node_.find(lhs);
    if (src_node_it == op_name2op_node_.end()) { return false; }
    const auto& dst_node_it = op_name2op_node_.find(rhs);
    if (dst_node_it == op_name2op_node_.end()) { return false; }
    return (src_node_it->second == dst_node_it->second)
           || IsDataOrCtrlReachable(src_node_it->second, dst_node_it->second);
  };
}

std::function<bool(const OpNode*, const OpNode*)> OpGraph::MakePredicatorIsDataOrCtrlReachable()
    const {
  auto _1 = std::placeholders::_1;
  auto _2 = std::placeholders::_2;
  return MakePredicatorIsReachable(DataOrCtrlSourceNodes(),
                                   std::bind(&OpGraph::ForEachDataAndCtrlInNode, this, _1, _2),
                                   std::bind(&OpGraph::ForEachDataAndCtrlOutNode, this, _1, _2));
}

std::list<OpNode*> OpGraph::DataOrCtrlSourceNodes() const {
  std::list<OpNode*> ret;
  ForEachNode([&](OpNode* op_node) {
    size_t in_edges_cnt = 0;
    ForEachDataAndCtrlInNode(op_node, [&](OpNode*) { ++in_edges_cnt; });
    if (in_edges_cnt == 0) { ret.push_back(op_node); }
  });
  return ret;
}

void OpGraph::DumpLogicalBlobDesc(Job* job) const {
  auto* helper = job->mutable_helper();
  ForEachNode([&](const OpNode* node) {
    for (const auto& obn : node->op().output_bns()) {
      const auto& lbi = node->op().BnInOp2Lbi(obn);
      node->LogicalBlobDesc4Lbi(lbi).ToProto(
          &(*helper->mutable_lbn2logical_blob_desc())[GenLogicalBlobName(lbi)]);
    }
  });
}

void OpGraph::DumpParallelDistributionSignature(Job* job) const {
  ForEachNode([&](const OpNode* node) -> void {
    (*job->mutable_job_parallel_view_conf()
          ->mutable_op_name2parallel_distribution_signature_conf())[node->op().op_name()] =
        *CHECK_JUST(node->op().parallel_distribution_signature());
    if (node->parallel_desc().hierarchy()->NumAxes() == 1) {
      (*job->mutable_job_parallel_view_conf()
            ->mutable_op_name2sbp_signature_conf())[node->op().op_name()] = node->sbp_signature();
    }
  });
}

void OpGraph::DumpArgSignature(Job* job) const {
  ForEachNode([&](const OpNode* node) {
    auto* op_arg_signature =
        &(*job->mutable_helper()->mutable_op_name2arg_signature())[node->op().op_name()];
    for (const auto& ibn : node->op().input_bns()) {
      const auto& lbi = node->op().BnInOp2Lbi(ibn);
      (*op_arg_signature->mutable_bn_in_op2lbi())[ibn] = lbi;
    }
    for (const auto& obn : node->op().output_bns()) {
      const auto& lbi = node->op().BnInOp2Lbi(obn);
      (*op_arg_signature->mutable_bn_in_op2lbi())[obn] = lbi;
    }
  });
}

Maybe<void> OpGraph::ForEachOpNode(const std::function<Maybe<void>(const OpNode&)>& DoEach) const {
  HashMap<LogicalBlobId, bool> visited;
  for (const auto& op_name : op_names_) {
    const OpNode& op_node = *op_name2op_node_.at(op_name);
    for (const auto& ibn : op_node.op().input_bns()) {
      const auto& lbi = op_node.op().BnInOp2Lbi(ibn);
      CHECK_OR_RETURN(visited[lbi]) << "input blob '" << ibn << "' is not defined\n"
                                    << lbi.DebugString() << "\n==== op_conf ====\n"
                                    << op_node.op().op_conf().DebugString();
    }
    for (const auto& obn : op_node.op().output_bns()) {
      const auto& lbi = op_node.op().BnInOp2Lbi(obn);
      CHECK_OR_RETURN(!visited[lbi]) << "output blob '" << obn << "' is defined\n"
                                     << lbi.DebugString() << "\n==== op_conf ====\n"
                                     << op_node.op().op_conf().DebugString();
      visited[lbi] = true;
    }
    JUST(DoEach(op_node));
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
