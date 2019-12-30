#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/operator/normal_model_update_op.h"

namespace oneflow {

namespace {

void UpdateSbpConf(const OpNode& op_node,
                   const HashMap<OpBlobArg, std::vector<OpBlobArg>>& oba2sbp_identical_obas,
                   SbpConf* sbp_conf) {
  auto* op_name2sbp_signature = sbp_conf->mutable_op_name2sbp_signature_conf();
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

OptInt64* OpNode::MutBatchAxis4Lbi(const LogicalBlobId& lbi) {
  CHECK_EQ(MutProducerOpNode4Lbi(lbi), this);
  return &lbi2batch_axis_[lbi];
}
const OptInt64& OpNode::BatchAxis4Lbi(const LogicalBlobId& lbi) const {
  return ProducerOpNode4Lbi(lbi).lbi2batch_axis_.at(lbi);
}

const SbpParallel& OpNode::SbpParallel4BnInOp(const std::string& bn_in_op) const {
  return sbp_signature_.bn_in_op2sbp_parallel().at(bn_in_op);
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
  return MutSrcNode4InputBnInOp(bn_in_op)->out_blob_time_shape();
}

const OpNode& OpNode::ProducerOpNode4BnInOp(const std::string& bn_in_op) const {
  if (ibns_.find(bn_in_op) != ibns_.end()) { return SrcNode4InputBnInOp(bn_in_op); }
  return *this;
}

const OpNode& OpNode::SrcNode4InputBnInOp(const std::string& bn_in_op) const {
  return *MutSrcNode4InputBnInOp(bn_in_op);
}

OpNode* OpNode::MutProducerOpNode4BnInOp(const std::string& bn_in_op) {
  if (ibns_.find(bn_in_op) != ibns_.end()) { return MutSrcNode4InputBnInOp(bn_in_op); }
  return this;
}

OpNode* OpNode::MutSrcNode4InputBnInOp(const std::string& bn_in_op) const {
  const LogicalBlobId& lbi = op().BnInOp2Lbi(bn_in_op);
  CHECK(ibns_.find(bn_in_op) != ibns_.end());
  return MutSrcNode4InputLbi(lbi);
}

OpNode* OpNode::MutProducerOpNode4Lbi(const LogicalBlobId& lbi) {
  OpNode* producer = MutSrcNode4InputLbi(lbi);
  if (producer == nullptr) { producer = this; }
  return producer;
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
    const BlobDesc& logical_blob_desc = ProducerOpNode4BnInOp(bn).LogicalBlobDesc4Lbi(lbi);
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
  CHECK_JUST(op().InferOutParallelDescIf(ParallelDesc4Obn, LogicalBlobDesc4Ibn, parallel_desc(),
                                         &sbp_signature()));
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

void OpGraph::Init(const Job& job) {
  InitNodes(job);
  ForEachNode(
      [&](OpNode* node) { CHECK(op_name2op_node_.emplace(node->op().op_name(), node).second); });
  InitEdges();
  InitProducerOpName2CtrlConsumerOpNames(job);
  CheckIsDAG();
  ForEachNode([](OpNode* node) { node->InitLbi2SourceNode(); });
  InferTimeShape();
  InferLogicalBlobDesc(job);
  ForEachEdge([](OpEdge* edge) { edge->InitDistributeHierarchyInfo(); });
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
      const auto& lbi2obn = producer_op_name2lbi2obn.at(pair.first);
      OpNode* producer = lbi2producer.at(lbis->at(0));
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
    OpNode* producer = op_node->MutSrcNode4InputBnInOp(ibn);
    const ParallelDesc* parallel_desc = &producer->parallel_desc();
    const BlobDesc* logical_blob_desc = &producer->LogicalBlobDesc4Lbi(lbi);
    const SbpParallel* sbp = &producer->SbpParallel4Lbi(lbi);
    const OptInt64* batch_axis = &producer->BatchAxis4Lbi(lbi);
    ibn2sbp_infer_hint.emplace(ibn,
                               SbpInferHint(parallel_desc, logical_blob_desc, sbp, batch_axis));
  }
  auto GetBatchAxis4Lbi = [&](const LogicalBlobId& lbi) -> const OptInt64& {
    return op_node->BatchAxis4Lbi(lbi);
  };
  CHECK_JUST(InferOpSbpSignature(op_node->op(), sbp_sig_conf, op_node->parallel_desc(),
                                 ibn2sbp_infer_hint, GetBatchAxis4Lbi,
                                 op_node->mut_sbp_signature()));
  op_node->InitLbi2SbpParallel();
}

void OpGraph::InferOpNodeLogicalBlobDesc(OpNode* op_node) const {
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
    CHECK_JUST(op_node->op().InferBlobDescsIf(BlobDesc4BnInOp, &parallel_ctx,
                                              &op_node->sbp_signature(), [](OpContext*) {}));
  }
  op_node->ConcatLogicalOutputBlobDesc();
}

const OpNode* OpGraph::OpNode4OpName(const std::string& op_name) const {
  const auto& op_node_it = op_name2op_node_.find(op_name);
  if (op_node_it == op_name2op_node_.end()) { return nullptr; }
  return op_node_it->second;
}

void OpGraph::InferLogicalBlobDesc(const Job& job) const {
  SbpConf sbp_conf(job.sbp_conf());
  HashMap<OpBlobArg, std::vector<OpBlobArg>> oba2sbp_identical_obas;
  for (const auto& pair : job.helper().identical_sbp_oba_pairs().pair()) {
    oba2sbp_identical_obas[pair.first()].push_back(pair.second());
    oba2sbp_identical_obas[pair.second()].push_back(pair.first());
  }
  TopoForEachNode([&](OpNode* op_node) {
    // infer batch_axis
    auto BatchAxis4BnInOp = [&](const std::string& bn) -> OptInt64* {
      return op_node->MutProducerOpNode4BnInOp(bn)->MutBatchAxis4Lbi(op_node->op().BnInOp2Lbi(bn));
    };
    auto LogicalBlobDesc4Ibn = [&](const std::string& ibn) -> const BlobDesc& {
      const auto& ibns = op_node->op().input_bns();
      CHECK(std::find(ibns.begin(), ibns.end(), ibn) != ibns.end());
      return op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi(ibn));
    };
    CHECK_JUST(op_node->op().InferBatchAxisIf(LogicalBlobDesc4Ibn, BatchAxis4BnInOp));
    // infer sbp_signature
    SbpSignature sbp_sig_conf;
    {
      const auto& op_name2sbp_sig_conf = sbp_conf.op_name2sbp_signature_conf();
      const auto& it = op_name2sbp_sig_conf.find(op_node->op().op_name());
      if (it != op_name2sbp_sig_conf.end()) { sbp_sig_conf = it->second; }
    }
    InferOpNodeSbpSignature(op_node, sbp_sig_conf);
    op_node->InferBlobParallelDesc();
    UpdateSbpConf(*op_node, oba2sbp_identical_obas, &sbp_conf);
    // infer logical_blob_desc
    InferOpNodeLogicalBlobDesc(op_node);
  });
  // fix sbp_signature
  {
    TopoForEachNode([&](OpNode* op_node) {
      if (op_node->op().op_conf().op_type_case() == OperatorConf::kCastConf) {
        if (op_node->out_edges().size() > 1) { return; }
        if (dynamic_cast<const NormalModelUpdtOp*>(&(op_node->SoleOutEdge()->dst_node()->op()))) {
          auto* bn2sbp = op_node->mut_sbp_signature()->mutable_bn_in_op2sbp_parallel();
          if (bn2sbp->at("out").has_partial_sum_parallel() && GlobalJobDesc().all_reduce_fp16()) {
            bn2sbp->at("in").mutable_broadcast_parallel();
            bn2sbp->at("out").mutable_broadcast_parallel();
          }
        }
      }
    });
  }
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
  return op_name2op_node_.at(GetOpNameKey(op_name, lbi))
      ->BatchAxis4Lbi(GetLogicalBlobIdKey(op_name, lbi))
      .has_value();
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

void OpGraph::DumpLogicalBlobDesc(JobBuilder* job_builder) const {
  auto* helper = job_builder->mutable_helper();
  ForEachNode([&](const OpNode* node) {
    for (const auto& obn : node->op().output_bns()) {
      const auto& lbi = node->op().BnInOp2Lbi(obn);
      node->LogicalBlobDesc4Lbi(lbi).ToProto(
          &(*helper->mutable_lbn2logical_blob_desc())[GenLogicalBlobName(lbi)]);
    }
  });
}

void OpGraph::DumpSbpSignature(JobBuilder* job_builder) const {
  ForEachNode([&](const OpNode* node) {
    (*job_builder->mutable_sbp_conf()->mutable_op_name2sbp_signature_conf())[node->op().op_name()] =
        node->sbp_signature();
  });
}

void OpGraph::DumpOpTimeShape(JobBuilder* job_builder) const {
  ForEachNode([&](OpNode* op_node) {
    auto* op_time_shape =
        &(*job_builder->mutable_helper()->mutable_op_name2op_time_shape())[op_node->op().op_name()];
    if (op_node->out_blob_time_shape() != nullptr) {
      op_node->out_blob_time_shape()->ToProto(op_time_shape->mutable_out_blob_time_shape());
    }
    const auto* in_blob_fastest_time_shape = op_node->GetInputBlobFastestTimeShape();
    if (in_blob_fastest_time_shape != nullptr) {
      in_blob_fastest_time_shape->ToProto(op_time_shape->mutable_in_blob_fastest_time_shape());
    }
  });
}

void OpGraph::DumpBatchAxisLbi(JobBuilder* job_builder) const {
  auto* lbn2batch_axis = job_builder->mutable_helper()->mutable_lbn2batch_axis();
  ForEachNode([&](OpNode* op_node) {
    for (const auto& obn : op_node->op().output_bns()) {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(obn);
      const auto& lbn = GenLogicalBlobName(lbi);
      const auto& pair = PbMapPair<std::string, OptInt64>(lbn, op_node->BatchAxis4Lbi(lbi));
      CHECK(lbn2batch_axis->insert(pair).second);
    }
  });
}

}  // namespace oneflow
