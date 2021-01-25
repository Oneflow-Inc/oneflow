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
#include "oneflow/core/operator/normal_model_update_op.h"
#include "oneflow/core/auto_parallel/sbp_constructor.h"

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
  if (src->IsTimeShapeIdentity() == false) { return false; }
  if (dst->IsTimeShapeIdentity() == false) { return false; }
  if (*src->GetInputOutputFastestTimeShape() != *dst->GetInputOutputFastestTimeShape()) {
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

Maybe<const OptInt64*> OpNode::BatchAxis4Lbi(const LogicalBlobId& lbi) const {
  const auto& op = ProducerOpNode4Lbi(lbi).op();
  return op.BatchAxis4BnInOp(*JUST(op.obn4lbi(lbi)));
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
      std::string dev_type;
      if (parallel_desc().device_type() == DeviceType::kCPU) {
        dev_type = "cpu";
      } else if (parallel_desc().device_type() == DeviceType::kGPU) {
        dev_type = "gpu";
      } else {
        UNIMPLEMENTED();
      }
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
    str += "\\n" + GetTimeShapeStr(*GetInputBlobFastestTimeShape(), "in_blob_time_shape");
  }
  str += "\\n" + GetTimeShapeStr(*out_blob_time_shape(), "out_blob_time_shape");
  for (const auto& ibn : op().input_bns()) {
    str += "\\n";
    auto producer_node = MutSrcNode4Ibn(ibn);
    str += "Pre Op:" + producer_node->op().op_name() + ": " + ibn;
    const SbpParallel& this_sbp_parallel = SbpParallel4BnInOp(ibn);
    if (this_sbp_parallel.has_split_parallel())
      str += " [S]" + std::to_string(this_sbp_parallel.split_parallel().axis()) + " ";
    if (this_sbp_parallel.has_broadcast_parallel()) str += " [B] ";
    if (this_sbp_parallel.has_partial_sum_parallel()) str += " [P] ";
  }
  for (const auto& ibn : op().output_bns()) {
    str += "\\n";
    str += "Out Op:" + ibn;
    const SbpParallel& this_sbp_parallel = SbpParallel4BnInOp(ibn);
    if (this_sbp_parallel.has_split_parallel())
      str += " [S]" + std::to_string(this_sbp_parallel.split_parallel().axis()) + " ";
    if (this_sbp_parallel.has_broadcast_parallel()) str += " [B] ";
    if (this_sbp_parallel.has_partial_sum_parallel()) str += " [P] ";
  }
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

const Shape* OpNode::out_blob_time_shape() const {
  const Shape* ret = out_blob_time_shape_.get();
  if (ret != nullptr && ret->elem_cnt() == 0) { return nullptr; }
  return ret;
}

Shape* OpNode::mut_out_blob_time_shape() {
  if (!out_blob_time_shape_) { out_blob_time_shape_.reset(new Shape()); }
  return out_blob_time_shape_.get();
}

const Shape* OpNode::GetInputBlobTimeShape(const std::string& bn_in_op) const {
  return MutSrcNode4Ibn(bn_in_op)->out_blob_time_shape();
}

const OpNode& OpNode::SrcNode4Ibn(const std::string& bn_in_op) const {
  return *MutSrcNode4Ibn(bn_in_op);
}

// mutable source node with specified input blob name
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
  const auto* in_shape = GetInputBlobFastestTimeShape();
  if (in_shape == nullptr) { return true; }
  const auto* out_shape = out_blob_time_shape();
  if (out_shape == nullptr) { return true; }
  return *in_shape == *out_shape;
}

const Shape* OpNode::GetInputBlobFastestTimeShape() const {
  return input_blob_fastest_time_shape_.get();
}

const Shape* OpNode::GetInputOutputFastestTimeShape() const {
  const Shape* in = GetInputBlobFastestTimeShape();
  const Shape* out = out_blob_time_shape();
  if (in == nullptr) { return out; }
  if (out == nullptr) { return in; }
  return in->elem_cnt() > out->elem_cnt() ? in : out;
}

void OpNode::ForEachSplitOrBroadcastBlobDesc(
    const BlobDesc& blob_desc, const SbpParallel& sbp_parallel,
    const std::function<void(const BlobDesc&)>& Handler) const {
  if (sbp_parallel.has_split_parallel()) {
    // split BlobDesc
    int32_t axis = sbp_parallel.split_parallel().axis();
    CHECK_GE(axis, 0);
    CHECK_LT(axis, blob_desc.shape().NumAxes());
    CHECK_GE(blob_desc.shape().At(axis), parallel_desc().parallel_num());
    BalancedSplitter bs(blob_desc.shape().At(axis), parallel_desc().parallel_num());
    BlobDesc sub_blob_desc(blob_desc);
    FOR_RANGE(int64_t, axis_parallel_id, 0, parallel_desc().parallel_num()) {
      sub_blob_desc.mut_shape().Set(axis, bs.At(axis_parallel_id).size());
      Handler(sub_blob_desc);
    }
  } else {
    CHECK(sbp_parallel.has_broadcast_parallel() || sbp_parallel.has_partial_sum_parallel());
    // broadcast BlobDesc
    FOR_RANGE(int64_t, axis_parallel_id, 0, parallel_desc().parallel_num()) { Handler(blob_desc); }
  }
}

void OpNode::ConcatBlobDesc(const ParallelDesc& blob_parallel_desc,
                            const std::vector<std::shared_ptr<BlobDesc>>& blob_descs,
                            const SbpParallel& sbp_parallel,
                            BlobDesc* concatenated_blob_desc) const {
  CHECK_EQ(blob_descs.size(), blob_parallel_desc.parallel_num());
  if (sbp_parallel.has_split_parallel()) {
    int32_t axis = sbp_parallel.split_parallel().axis();
    // concat BlobDesc
    CHECK_GE(axis, 0);
    CHECK_LT(axis, blob_descs.at(0)->shape().NumAxes());
    int64_t logical_blob_axis_dim = 0;
    for (const auto& blob_desc : blob_descs) {
      logical_blob_axis_dim += blob_desc->shape().At(axis);
    }
    CHECK_GE(logical_blob_axis_dim, blob_parallel_desc.parallel_num());
    BalancedSplitter bs(logical_blob_axis_dim, blob_parallel_desc.parallel_num());
    std::vector<std::unique_ptr<BlobDesc>> same_blob_descs(blob_descs.size());
    FOR_RANGE(int64_t, axis_parallel_id, 0, blob_parallel_desc.parallel_num()) {
      CHECK_EQ(bs.At(axis_parallel_id).size(), blob_descs.at(axis_parallel_id)->shape().At(axis));
      same_blob_descs.at(axis_parallel_id).reset(new BlobDesc(*blob_descs.at(axis_parallel_id)));
      same_blob_descs.at(axis_parallel_id)->mut_shape().Set(axis, logical_blob_axis_dim);
    }
    FOR_RANGE(int64_t, i, 1, same_blob_descs.size()) {
      CHECK(*same_blob_descs.at(i) == *same_blob_descs.at(0));
    }
    concatenated_blob_desc->CopyAllFrom(*same_blob_descs.at(0));
  } else {
    FOR_RANGE(int64_t, i, 1, blob_descs.size()) { CHECK(*blob_descs.at(i) == *blob_descs.at(0)); }
    // select first BlobDesc
    concatenated_blob_desc->CopyAllFrom(*blob_descs.at(0));
  }
}

int64_t OpNode::GetAxisParallelNum(
    const std::function<void(bool*, int32_t*, int64_t*)>& GetAxisParallelInfo) const {
  bool is_split = false;
  int32_t axis = -1;
  int64_t axis_parallel_num = 0;
  GetAxisParallelInfo(&is_split, &axis, &axis_parallel_num);
  return axis_parallel_num;
}

void OpNode::SplitLogicalInputBlobDesc() {
  for (const std::string& bn : op().input_bns()) {
    const LogicalBlobId& lbi = op().BnInOp2Lbi(bn);
    const BlobDesc& logical_blob_desc = SrcNode4Ibn(bn).LogicalBlobDesc4Lbi(lbi);
    CHECK_NE(logical_blob_desc.data_type(), DataType::kInvalidDataType);
    const SbpParallel& sbp_parallel = SbpParallel4BnInOp(bn);
    ForEachSplitOrBroadcastBlobDesc(
        logical_blob_desc, sbp_parallel, [&](const BlobDesc& blob_desc) {
          bn2parallel_id2blob_desc_[bn].emplace_back(new BlobDesc(blob_desc));
          CHECK_NE(bn2parallel_id2blob_desc_[bn].back()->data_type(), DataType::kInvalidDataType);
        });
    CHECK_EQ(bn2parallel_id2blob_desc_.at(bn).size(), parallel_desc().parallel_num());
  }
}

void OpNode::ConcatLogicalOutputBlobDesc() {
  for (const std::string& obn : op().output_bns()) {
    const ParallelDesc& blob_parallel_desc = BlobParallelDesc4Obn(obn);
    const LogicalBlobId& lbi = op().BnInOp2Lbi(obn);
    const SbpParallel& sbp_parallel = SbpParallel4BnInOp(obn);
    std::vector<std::shared_ptr<BlobDesc>> paralleled_blob_descs;
    for (const auto& blob_desc : bn2parallel_id2blob_desc_.at(obn)) {
      if (blob_desc) { paralleled_blob_descs.push_back(blob_desc); }
    }
    ConcatBlobDesc(blob_parallel_desc, paralleled_blob_descs, sbp_parallel,
                   MutLogicalBlobDesc4Lbi(lbi));
  }
}

void OpNode::CheckBlobDescs(const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  return;
  int64_t parallel_id = parallel_ctx->parallel_id();
  auto Check = [&](const std::string& bn) {
    if (bn2parallel_id2blob_desc_.find(bn) == bn2parallel_id2blob_desc_.end()) { return; }
    CHECK_EQ(parallel_ctx->parallel_num(), bn2parallel_id2blob_desc_.at(bn).size());
    const BlobDesc& blob_desc_from_exec_graph = *GetBlobDesc4BnInOp(bn);
    const BlobDesc& blob_desc_from_op_graph = *bn2parallel_id2blob_desc_.at(bn).at(parallel_id);
    CHECK_EQ(blob_desc_from_exec_graph.shape(), blob_desc_from_op_graph.shape());
    CHECK_EQ(blob_desc_from_exec_graph.data_type(), blob_desc_from_op_graph.data_type());
  };
  for (const std::string& bn : op().input_bns()) { Check(bn); }
  for (const std::string& bn : op().output_bns()) { Check(bn); }
  for (const std::string& bn : op().tmp_bns()) { Check(bn); }
  for (const std::string& bn : op().const_buf_bns()) { Check(bn); }
}

const ParallelDesc& OpNode::BlobParallelDesc4Obn(const std::string& obn) const {
  return obn2blob_parallel_desc_.at(obn);
}

void OpNode::InferBlobParallelDesc() {
  auto ParallelDesc4Obn = [&](const std::string& obn) -> ParallelDesc* {
    auto iter = obn2blob_parallel_desc_.find(obn);
    if (iter == obn2blob_parallel_desc_.end()) {
      iter = obn2blob_parallel_desc_.emplace(obn, parallel_desc()).first;
    }
    return &iter->second;
  };
  auto LogicalBlobDesc4Ibn = [&](const std::string& ibn) -> const BlobDesc* {
    return &LogicalBlobDesc4Lbi(op().BnInOp2Lbi(ibn));
  };
  CHECK_JUST(op().InferOutParallelDescIf(ParallelDesc4Obn, LogicalBlobDesc4Ibn, parallel_desc()));
}

void OpNode::InitLbi2SourceNode() {
  for (OpEdge* edge : in_edges()) {
    for (const LogicalBlobId& lbi : edge->lbis()) {
      CHECK(lbi2source_node_.emplace(lbi, edge->src_node()).second);
    }
  }
}

void OpNode::InitInputBlobFastestTimeShape() {
  const Shape* fastest = nullptr;
  for (OpEdge* edge : in_edges()) {
    const Shape* shape = edge->src_node()->out_blob_time_shape();
    if (fastest == nullptr || shape->elem_cnt() > fastest->elem_cnt()) { fastest = shape; }
  }
  for (OpEdge* edge : in_edges()) {
    CHECK_EQ(fastest->elem_cnt() % edge->src_node()->out_blob_time_shape()->elem_cnt(), 0);
  }
  if (fastest != nullptr) { input_blob_fastest_time_shape_.reset(new Shape(fastest->dim_vec())); }
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

void OpNode::UpdateLbi2SbpParallel() {
  const auto Update = [&](const PbRpf<std::string>& bns) {
    for (const auto& bn : bns) {
      const LogicalBlobId& lbi = op().BnInOp2Lbi(bn);
      const SbpParallel& sbp_parallel = SbpParallel4BnInOp(bn);
      lbi2sbp_parallel_[lbi] = sbp_parallel;
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
  // For each op node, build connection between output blob names and logical blob ids.
  ForEachNode([&](OpNode* op_node) {
    for (const auto& obn : op_node->op().output_bns()) {
      const auto& lbi = op_node->op().BnInOp2Lbi(obn);
      CHECK(lbi2producer.emplace(lbi, op_node).second);
      auto& lbi2obn = producer_op_name2lbi2obn[op_node->op().op_name()];
      if (!lbi2obn) { lbi2obn.reset(new HashMap<LogicalBlobId, std::string>()); }
      CHECK(lbi2obn->emplace(lbi, obn).second);
    }
  });
  // For each op node, build connection between op names and logical blob ids list, and build
  // connection between logical blob ids and input blob names.
  ForEachNode([&](OpNode* op_node) {
    HashMap<std::string, HashSet<LogicalBlobId>> producer_op_name2lbis;
    std::shared_ptr<HashMap<LogicalBlobId, std::vector<std::string>>> consumer_lbi2ibns(
        new HashMap<LogicalBlobId, std::vector<std::string>>);
    for (const auto& ibn : op_node->op().input_bns()) {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
      producer_op_name2lbis[lbi.op_name()].insert(lbi);
      (*consumer_lbi2ibns)[lbi].push_back(ibn);
    }
    // Build edges use the connections computed above.
    for (const auto& pair : producer_op_name2lbis) {
      std::shared_ptr<std::vector<LogicalBlobId>> lbis(
          new std::vector<LogicalBlobId>({pair.second.begin(), pair.second.end()}));
      const auto it = producer_op_name2lbi2obn.find(pair.first);
      CHECK(it != producer_op_name2lbi2obn.end()) << "producer_op_name: " << pair.first;
      const auto& lbi2obn = it->second;
      OpNode* producer = lbi2producer.at(lbis->at(0));
      // Connect current node with an upstream node
      Connect(producer, NewEdge(lbis, lbi2obn, consumer_lbi2ibns), op_node);
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
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_id(0);
    parallel_ctx.set_parallel_num(op_node->parallel_desc().parallel_num());
    auto GetInputBlobTimeShape = [&](const std::string& bn_in_op) {
      return op_node->GetInputBlobTimeShape(bn_in_op);
    };
    op_node->InitInputBlobFastestTimeShape();
    CHECK_JUST(op_node->op().InferOutputBlobTimeShapeIf(GetInputBlobTimeShape, &parallel_ctx,
                                                        op_node->mut_out_blob_time_shape()));
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
    const OptInt64* batch_axis = CHECK_JUST(producer->BatchAxis4Lbi(lbi));
    ibn2sbp_infer_hint.emplace(ibn,
                               SbpInferHint(parallel_desc, logical_blob_desc, sbp, batch_axis));
  }
  const auto& BatchAxis4BnInOp = [&](const std::string& bn_in_op) -> Maybe<const OptInt64*> {
    return op_node->op().BatchAxis4BnInOp(bn_in_op);
  };
  CHECK_JUST(InferOpSbpSignature(op_node->mut_op(), sbp_sig_conf, op_node->parallel_desc(),
                                 ibn2sbp_infer_hint, BatchAxis4BnInOp));
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

Maybe<void> OpGraph::InferOpNodeLogicalBlobDesc(OpNode* op_node) const {
  auto* bn2parallel_id2blob_desc = op_node->mut_bn2parallel_id2blob_desc();
  op_node->SplitLogicalInputBlobDesc();
  int64_t parallel_num = op_node->parallel_desc().parallel_num();
  const auto& input_bns = op_node->op().input_bns();
  FOR_RANGE(int64_t, parallel_id, 0, parallel_num) {
    auto BlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
      if (std::find(input_bns.begin(), input_bns.end(), bn) != input_bns.end()) {
        CHECK(bn2parallel_id2blob_desc->find(bn) != bn2parallel_id2blob_desc->end());
        CHECK_EQ(bn2parallel_id2blob_desc->at(bn).size(), parallel_num);
      } else if (bn2parallel_id2blob_desc->find(bn) == bn2parallel_id2blob_desc->end()) {
        (*bn2parallel_id2blob_desc)[bn].resize(parallel_num);
      } else {
        CHECK_EQ(bn2parallel_id2blob_desc->at(bn).size(), parallel_num);
      }
      auto* blob_desc = &bn2parallel_id2blob_desc->at(bn).at(parallel_id);
      if (!*blob_desc) { blob_desc->reset(new BlobDesc(GlobalJobDesc().DefaultDataType())); }
      return blob_desc->get();
    };
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_id(parallel_id);
    parallel_ctx.set_parallel_num(parallel_num);
    JUST(op_node->op().InferOutBlobDescsIf(BlobDesc4BnInOp, &parallel_ctx,
                                           &op_node->sbp_signature(), [](OpContext*) {}));
  }
  op_node->ConcatLogicalOutputBlobDesc();
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
    // Infer ParallelSignature
    JUST(op_node->mut_op()->InferParallelSignatureIf());
    // Infer batch_axis
    const auto& BatchAxis4Ibn = [&](const std::string& ibn) -> Maybe<const OptInt64*> {
      const auto& lbi = op_node->op().BnInOp2Lbi(ibn);
      const auto* producer = op_node->MutSrcNode4InputLbi(lbi);
      CHECK_NOTNULL_OR_RETURN(producer);
      return producer->op().BatchAxis4BnInOp(*JUST(producer->op().obn4lbi(lbi)));
    };
    const auto& LogicalBlobDesc4Ibn = [&](const std::string& ibn) -> const BlobDesc& {
      const auto& ibns = op_node->op().input_bns();
      CHECK(std::find(ibns.begin(), ibns.end(), ibn) != ibns.end());
      return op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(ibn));
    };
    JUST(op_node->mut_op()->InferBatchAxisIf(LogicalBlobDesc4Ibn, BatchAxis4Ibn));
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
#ifndef ENABLE_AUTO_PARALLEL
    // SbpConstructor: Do not update to job because it will limit sbp_node to choose condidate
    UpdateJobParallelViewConf(*op_node, oba2sbp_identical_obas, &job_parallel_view_conf);
#endif
    // Infer logical_blob_desc
    JUST(InferOpNodeLogicalBlobDesc(op_node));
    // Fill logical blob_desc signature.
    JUST(op_node->mut_op()->FillLogicalBlobDescSignature(
        [&](const std::string& bn_in_op) -> Maybe<const BlobDesc&> {
          return op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(bn_in_op));
        }));

    // // test debug
    // std::cout << op_node->op().op_name() << " (^_^):" << std::endl;
    // for (const auto& ibn : op_node->op().input_bns()) {
    //   auto producer_node = op_node->MutSrcNode4Ibn(ibn);
    //   std::cout << "Pre Op:" << producer_node->op().op_name() << ": " << ibn;
    //   const SbpParallel& this_sbp_parallel = op_node->SbpParallel4BnInOp(ibn);
    //   if (this_sbp_parallel.has_split_parallel()) std::cout << " S" <<
    //   this_sbp_parallel.split_parallel().axis(); if (this_sbp_parallel.has_broadcast_parallel())
    //   std::cout << " B"; if (this_sbp_parallel.has_partial_sum_parallel()) std::cout << " P";
    //   std::cout << std::endl;
    //   /* auto blob_desc = op_node->mut_bn2parallel_id2blob_desc()->at(ibn).at(0); */
    //   /* std::cout << " shape:" << blob_desc->shape().DebugStr() << std::endl; */
    // }
    // for (const auto& ibn : op_node->op().output_bns()) {
    //   std::cout << "Out Op:" << ibn;
    //   const SbpParallel& this_sbp_parallel = op_node->SbpParallel4BnInOp(ibn);
    //   if (this_sbp_parallel.has_split_parallel()) std::cout << " S" <<
    //   this_sbp_parallel.split_parallel().axis(); if (this_sbp_parallel.has_broadcast_parallel())
    //   std::cout << " B"; if (this_sbp_parallel.has_partial_sum_parallel()) std::cout << " P";
    //   std::cout << std::endl;
    //   /* auto blob_desc = op_node->mut_bn2parallel_id2blob_desc()->at(ibn).at(0); */
    //   /* std::cout << " shape:" << blob_desc->shape().DebugStr() << std::endl; */
    // }
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

BalancedSplitter OpGraph::GetBalancedSplitter(const std::string& op_name,
                                              const LogicalBlobId& lbi) const {
  OpNode* op_node = op_name2op_node_.at(GetOpNameKey(op_name, lbi));
  const SbpParallel& sbp_parallel = GetSbpParallel(op_name, lbi);
  CHECK(sbp_parallel.has_split_parallel());
  int64_t split_num = GetSplitNum(op_name, lbi);
  CHECK_GE(split_num, op_node->parallel_desc().parallel_num());
  return BalancedSplitter(split_num, op_node->parallel_desc().parallel_num());
}

int32_t OpGraph::GetModelSplitAxis(const std::string& op_name, const LogicalBlobId& lbi) const {
  const SbpParallel& sbp_parallel = GetSbpParallel(op_name, lbi);
  CHECK(sbp_parallel.has_split_parallel());
  return sbp_parallel.split_parallel().axis();
}

int64_t OpGraph::GetSplitNum(const std::string& op_name, const LogicalBlobId& lbi) const {
  OpNode* op_node = op_name2op_node_.at(GetOpNameKey(op_name, lbi));
  const LogicalBlobId& lbi_key = GetLogicalBlobIdKey(op_name, lbi);
  const SbpParallel& sbp_parallel = op_node->SbpParallel4Lbi(lbi_key);
  CHECK(sbp_parallel.has_split_parallel());
  return op_node->LogicalBlobDesc4Lbi(lbi_key).shape().At(sbp_parallel.split_parallel().axis());
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

bool OpGraph::IsBatchAxisBlob(const std::string& op_name, const LogicalBlobId& lbi) const {
  const auto& op = *op_name2op_node_.at(GetOpNameKey(op_name, lbi));
  const auto& opt_int64 = *CHECK_JUST(op.BatchAxis4Lbi(GetLogicalBlobIdKey(op_name, lbi)));
  return opt_int64.has_value();
}

void OpGraph::CheckBlobDescs(const std::string& op_name,
                             const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  if (op_name2op_node_.find(op_name) == op_name2op_node_.end()) { return; }
  op_name2op_node_.at(op_name)->CheckBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
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

std::function<const BlobDesc&(const LogicalBlobId&)> OpGraph::MakeGetterBlobDesc4ModelLbi() const {
  HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>> lbi2unparalleled_blob_desc;
  DataType dtype = GlobalJobDesc().DefaultDataType();
  TopoForEachNode([&](OpNode* op_node) {
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_id(0);
    parallel_ctx.set_parallel_num(1);
    SbpSignature sbp_signature;
    for (const auto& ibn : op_node->op().input_bns()) {
      (*sbp_signature.mutable_bn_in_op2sbp_parallel())[ibn].mutable_split_parallel()->set_axis(0);
    }
    for (const auto& obn : op_node->op().output_bns()) {
      (*sbp_signature.mutable_bn_in_op2sbp_parallel())[obn].mutable_split_parallel()->set_axis(0);
    }
    auto MutUnparalleledBlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
      const auto& lbi = op_node->op().BnInOp2Lbi(bn);
      auto it = lbi2unparalleled_blob_desc.find(lbi);
      if (it == lbi2unparalleled_blob_desc.end()) {
        auto& blob_desc = lbi2unparalleled_blob_desc[lbi];
        blob_desc.reset(new BlobDesc(dtype));
        return blob_desc.get();
      }
      return it->second.get();
    };
    // the real important data we want to get is:
    // a) model blobs' byte size;
    // b) number of axes of blobs' body shape;
    CHECK_JUST(op_node->op().InferOutBlobDescsIf(MutUnparalleledBlobDesc4BnInOp, &parallel_ctx,
                                                 &sbp_signature, [](OpContext*) {}));
  });
  auto model_lbi2blob_desc = std::make_shared<HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>>>();
  ForEachNode([&](OpNode* op_node) {
    for (const std::string& tmp_bn : op_node->op().tmp_bns()) {
      const auto& lbi = op_node->op().BnInOp2Lbi(tmp_bn);
      const auto& iter = lbi2unparalleled_blob_desc.find(lbi);
      if (iter == lbi2unparalleled_blob_desc.end()) { continue; }
      CHECK(model_lbi2blob_desc->emplace(lbi, std::move(iter->second)).second);
    }
  });
  return [model_lbi2blob_desc](const LogicalBlobId& model_lbi) -> const BlobDesc& {
    return *model_lbi2blob_desc->at(model_lbi);
  };
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

void OpGraph::DumpOpTimeShape(Job* job) const {
  ForEachNode([&](OpNode* op_node) {
    auto* op_time_shape =
        &(*job->mutable_helper()->mutable_op_name2op_time_shape())[op_node->op().op_name()];
    if (op_node->out_blob_time_shape() != nullptr) {
      op_node->out_blob_time_shape()->ToProto(op_time_shape->mutable_out_blob_time_shape());
    }
    const auto* in_blob_fastest_time_shape = op_node->GetInputBlobFastestTimeShape();
    if (in_blob_fastest_time_shape != nullptr) {
      in_blob_fastest_time_shape->ToProto(op_time_shape->mutable_in_blob_fastest_time_shape());
    }
  });
}

void OpGraph::DumpBatchAxisLbi(Job* job) const {
  auto* lbn2batch_axis = job->mutable_helper()->mutable_lbn2batch_axis();
  ForEachNode([&](OpNode* op_node) {
    for (const auto& obn : op_node->op().output_bns()) {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(obn);
      const auto& lbn = GenLogicalBlobName(lbi);
      const auto& batch_axis = *CHECK_JUST(op_node->BatchAxis4Lbi(lbi));
      const auto& pair = PbMapPair<std::string, OptInt64>(lbn, batch_axis);
      CHECK(lbn2batch_axis->insert(pair).first->second == batch_axis);
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
