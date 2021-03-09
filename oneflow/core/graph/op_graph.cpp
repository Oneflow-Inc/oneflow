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

namespace {

void UpdateJobParallelViewConf(
    const OpNode& op_node, const HashMap<OpBlobArg, std::vector<OpBlobArg>>& oba2sbp_identical_obas,
    JobParallelViewConf* job_parallel_view_conf) {
  auto* op_name2sbp_signature = job_parallel_view_conf->mutable_op_name2sbp_signature_conf();
  auto Update = [&](const std::string& bn) {
    const auto& sbp_parallel = op_node.sbp_signature().bn_in_op2sbp_parallel().at(bn);
    const OpBlobArg& oba = GenOpBlobArg(op_node.op().op_name(), bn);
    auto iter = oba2sbp_identical_obas.find(oba);
    if (iter == oba2sbp_identical_obas.end()) { return; }
    for (const auto& identical_obas : iter->second) {
      auto* sbp_signature = &(*op_name2sbp_signature)[identical_obas.op_name()];
      auto iter = sbp_signature->mutable_bn_in_op2sbp_parallel()->find(identical_obas.bn_in_op());
      if (iter == sbp_signature->mutable_bn_in_op2sbp_parallel()->end()) {
        CHECK(iter->second == sbp_parallel);
      } else {
        iter->second = sbp_parallel;
      }
    }
  };
  for (const auto& ibn : op_node.op().input_bns()) { Update(ibn); }
  for (const auto& obn : op_node.op().output_bns()) { Update(obn); }
}

}  // namespace

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
    const SbpParallel& obn_sbp = src->SbpParallel4BnInOp(lbi2obn().at(lbi));
    for (const std::string& ibn : lbi2ibns().at(lbi)) {
      const SbpParallel& ibn_sbp = dst->SbpParallel4BnInOp(ibn);
      if (obn_sbp != ibn_sbp) { return false; }
    }
  }
  return true;
}

const SbpParallel& OpNode::SbpParallel4BnInOp(const std::string& bn_in_op) const {
  return *CHECK_JUST(op().SbpParallel4BnInOp(bn_in_op));
}

const SbpParallel& OpNode::SbpParallel4Lbi(const LogicalBlobId& lbi) const {
  auto it = lbi2sbp_parallel_.find(lbi);
  CHECK(it != lbi2sbp_parallel_.end());
  return it->second;
}

std::string OpNode::VisualStr() const {
  std::string str = op().op_name();
  {
    for (int64_t machine_id : parallel_desc().sorted_machine_ids()) {
      const char* dev_type = CHECK_JUST(DeviceTag4DeviceType(parallel_desc().device_type()));

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
  return *ProducerOpNode4Lbi(lbi).lbi2logical_blob_desc_.at(lbi);
}

BlobDesc* OpNode::MutLogicalBlobDesc4Lbi(const LogicalBlobId& lbi) {
  CHECK_EQ(lbi.op_name(), op().op_name());
  if (lbi2logical_blob_desc_.find(lbi) == lbi2logical_blob_desc_.end()) {
    lbi2logical_blob_desc_[lbi].reset(new BlobDesc(GlobalJobDesc().DefaultDataType()));
  }
  return lbi2logical_blob_desc_.at(lbi).get();
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

void OpNode::InitLbi2SbpParallel() {
  const auto Update = [&](const PbRpf<std::string>& bns) {
    for (const auto& bn : bns) {
      const LogicalBlobId& lbi = op().BnInOp2Lbi(bn);
      const SbpParallel& sbp_parallel = SbpParallel4BnInOp(bn);
      auto it = lbi2sbp_parallel_.find(lbi);
      if (it == lbi2sbp_parallel_.end()) {
        lbi2sbp_parallel_[lbi] = sbp_parallel;
      } else {
        CHECK(it->second == sbp_parallel);
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

void OpGraph::InitNodes(const Job& job) {
  auto ParallelConf4OpName = MakeGetterParallelConf4OpName(job.placement());
  for (const auto& op_conf : job.net().op()) {
    op_names_.push_back(op_conf.name());
    OpNode* node = new OpNode(ParallelDesc(*ParallelConf4OpName(op_conf.name())), op_conf);
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
    for (const auto& ibn : op_node->op().input_bns()) {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
      producer_op_name2lbis[lbi.op_name()].insert(lbi);
      (*consumer_lbi2ibns)[lbi].push_back(ibn);
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

void OpGraph::InferOpNodeSbpSignature(OpNode* op_node, const SbpSignature& sbp_sig_conf) const {
  HashMap<std::string, SbpInferHint> ibn2sbp_infer_hint;
  for (const std::string& ibn : op_node->op().input_bns()) {
    const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
    OpNode* producer = op_node->MutSrcNode4Ibn(ibn);
    const ParallelDesc* parallel_desc = &producer->parallel_desc();
    const BlobDesc* logical_blob_desc = &producer->LogicalBlobDesc4Lbi(lbi);
    const SbpParallel* sbp = &producer->SbpParallel4Lbi(lbi);
    ibn2sbp_infer_hint.emplace(ibn, SbpInferHint(parallel_desc, logical_blob_desc, sbp));
  }
  CHECK_JUST(InferOpSbpSignature(op_node->mut_op(), sbp_sig_conf, op_node->parallel_desc(),
                                 ibn2sbp_infer_hint));
  op_node->InitLbi2SbpParallel();
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
  HashMap<OpBlobArg, std::vector<OpBlobArg>> oba2sbp_identical_obas;
  for (const auto& pair : job.helper().identical_sbp_oba_pairs().pair()) {
    oba2sbp_identical_obas[pair.first()].push_back(pair.second());
    oba2sbp_identical_obas[pair.second()].push_back(pair.first());
  }
  JUST(TopoForEachNodeWithErrorCaptured([&](OpNode* op_node) -> Maybe<void> {
    auto LogicalBlobDesc4BnInOp = [&](const std::string& bn) -> const BlobDesc& {
      return op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(bn));
    };
    JUST(op_node->mut_op()->FillOpParallelDesc(op_node->parallel_desc()));
    JUST(op_node->mut_op()->FillLogicalInBlobDesc(LogicalBlobDesc4BnInOp));
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
    // Infer sbp_signature
    SbpSignature sbp_sig_conf;
    {
      const auto& op_name2sbp_sig_conf = job_parallel_view_conf.op_name2sbp_signature_conf();
      const auto& iter = op_name2sbp_sig_conf.find(op_node->op().op_name());
      if (iter != op_name2sbp_sig_conf.end()) { sbp_sig_conf = iter->second; }
    }
    InferOpNodeSbpSignature(op_node, sbp_sig_conf);
    op_node->InferBlobParallelDesc();
    UpdateJobParallelViewConf(*op_node, oba2sbp_identical_obas, &job_parallel_view_conf);
    JUST(op_node->mut_op()->InferLogicalOutBlobDescsIf());
    for (const auto& bn : op_node->op().output_bns()) {
      *op_node->MutLogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(bn)) =
          *JUST(op_node->op().GetLogicalBlobDesc4Obn(bn));
    }
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

DataType OpGraph::GetBlobDataType(const LogicalBlobId& lbi) const {
  return op_name2op_node_.at(lbi.op_name())
      ->LogicalBlobDesc4Lbi(GetLogicalBlobIdKey(lbi.op_name(), lbi))
      .data_type();
}

const BlobDesc& OpGraph::GetLogicalBlobDesc(const LogicalBlobId& lbi) const {
  return op_name2op_node_.at(lbi.op_name())
      ->LogicalBlobDesc4Lbi(GetLogicalBlobIdKey(lbi.op_name(), lbi));
}

void OpGraph::ForEachChainFamily(
    const std::function<void(const HashSet<OpNode*>&)>& Handler) const {
  auto ForEachConnectedWithSameSbp7ParallelDesc7TimeShape =
      [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
        for (OpEdge* edge : node->in_edges()) {
          if (edge->is_strict_121()) { Handler(edge->src_node()); }
        }
        for (OpEdge* edge : node->out_edges()) {
          if (edge->is_strict_121()) { Handler(edge->dst_node()); }
        }
      };
  ForEachConnectedComponent(ForEachConnectedWithSameSbp7ParallelDesc7TimeShape, Handler);
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

void OpGraph::DumpSbpSignature(Job* job) const {
  ForEachNode([&](const OpNode* node) {
    (*job->mutable_job_parallel_view_conf()
          ->mutable_op_name2sbp_signature_conf())[node->op().op_name()] = node->sbp_signature();
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
