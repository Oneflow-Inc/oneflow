#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

std::string OpEdge::VisualStr() const {
  std::string str;
  int32_t idx = 0;
  for (const LogicalBlobId& lbi : lbis_) {
    if (idx++ > 0) { str += "\\n"; }
    str += lbi.blob_name() + ":";
    str += src_node()->NoParallelBlobDesc4Lbi(lbi).shape().ToString();
  }
  return str;
}

bool* OpNode::MutIsModelBlob4Lbi(const LogicalBlobId& lbi) {
  CHECK_EQ(ProducerOpNode4Lbi(lbi), this);
  return &lbi2is_model_blob_[lbi];
}
bool OpNode::IsModelBlob4Lbi(const LogicalBlobId& lbi) const {
  return ProducerOpNode4Lbi(lbi)->lbi2is_model_blob_.at(lbi);
}

const SbpParallel& OpNode::SbpParallel4Lbi(const LogicalBlobId& lbi) const {
  return lbi2sbp_parallel_.at(lbi);
}

SbpParallel* OpNode::MutSbpParallel4Lbi(const LogicalBlobId& lbi) {
  return &lbi2sbp_parallel_[lbi];
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
    str += "\\n" + GetTimeShapeStr(*GetInputBlobTimeShape(), "in_blob_time_shape");
  }
  str += "\\n" + GetTimeShapeStr(out_blob_time_shape(), "out_blob_time_shape");
  return str;
}

const BlobDesc& OpNode::NoParallelBlobDesc4Lbi(const LogicalBlobId& lbi) const {
  return lbi2no_parallel_blob_desc_.at(lbi);
}

const BlobDesc& OpNode::LogicalBlobDesc4Lbi(const LogicalBlobId& lbi) const {
  return lbi2logical_blob_desc_.at(lbi);
}

BlobDesc* OpNode::MutNoParallelBlobDesc(const LogicalBlobId& lbi) {
  CHECK_EQ(lbi.op_name(), op().op_name());
  return &lbi2no_parallel_blob_desc_[lbi];
}

BlobDesc* OpNode::MutLogicalBlobDesc4Lbi(const LogicalBlobId& lbi) {
  CHECK_EQ(lbi.op_name(), op().op_name());
  return &lbi2logical_blob_desc_[lbi];
}

BlobDesc* OpNode::NoParallelBlobDesc4BnInOp(const std::string& bn_in_op) {
  return ProducerOpNode4BnInOp(bn_in_op)->MutNoParallelBlobDesc(op().BnInOp2Lbi(bn_in_op));
}

const Shape* OpNode::GetInputBlobTimeShape(const std::string& bn_in_op) const {
  return &SrcNode4InputBnInOp(bn_in_op)->out_blob_time_shape();
}

OpNode* OpNode::ProducerOpNode4BnInOp(const std::string& bn_in_op) {
  if (ibns_.find(bn_in_op) != ibns_.end()) { return SrcNode4InputBnInOp(bn_in_op); }
  return this;
}

OpNode* OpNode::SrcNode4InputBnInOp(const std::string& bn_in_op) const {
  const LogicalBlobId& lbi = op().BnInOp2Lbi(bn_in_op);
  CHECK(ibns_.find(bn_in_op) != ibns_.end());
  return SrcNode4InputLbi(lbi);
}

OpNode* OpNode::ProducerOpNode4Lbi(const LogicalBlobId& lbi) {
  OpNode* producer = SrcNode4InputLbi(lbi);
  if (producer == nullptr) { producer = this; }
  return producer;
}

const OpNode* OpNode::ProducerOpNode4Lbi(const LogicalBlobId& lbi) const {
  const OpNode* producer = SrcNode4InputLbi(lbi);
  if (producer == nullptr) { producer = this; }
  return producer;
}

OpNode* OpNode::SrcNode4InputLbi(const LogicalBlobId& lbi) const {
  for (OpEdge* edge : in_edges()) {
    for (const LogicalBlobId& edge_lbi : edge->lbis()) {
      if (lbi == edge_lbi) { return edge->src_node(); }
    }
  }
  return nullptr;
}

const Shape* OpNode::GetInputBlobTimeShape() const {
  if (in_edges().empty()) { UNIMPLEMENTED(); }
  OpNode* first_input = (*in_edges().begin())->src_node();
  for (OpEdge* edge : in_edges()) {
    CHECK_EQ(first_input->out_blob_time_shape(), edge->src_node()->out_blob_time_shape());
  }
  return &first_input->out_blob_time_shape();
}

void OpNode::ForEachParallelBlobDesc(const BlobDesc& blob_desc, const SbpParallel& sbp_parallel,
                                     const std::function<void(const BlobDesc&)>& Handler) const {
  if (sbp_parallel.has_split_parallel()) {
    // split BlobDesc
    int32_t axis = sbp_parallel.split_parallel().axis();
    CHECK_GE(axis, 0);
    CHECK_LT(axis, blob_desc.shape().NumAxes());
    CHECK_GE(blob_desc.shape().At(axis), parallel_desc().parallel_num());
    BalancedSplitter bs(blob_desc.shape().At(axis), parallel_desc().parallel_num());
    FOR_RANGE(int64_t, axis_parallel_id, 0, parallel_desc().parallel_num()) {
      BlobDesc sub_blob_desc(blob_desc);
      sub_blob_desc.mut_shape().Set(axis, bs.At(axis_parallel_id).size());
      Handler(sub_blob_desc);
    }
  } else {
    CHECK(sbp_parallel.has_broadcast_parallel() || sbp_parallel.has_partial_sum_parallel());
    // broadcast BlobDesc
    FOR_RANGE(int64_t, axis_parallel_id, 0, parallel_desc().parallel_num()) { Handler(blob_desc); }
  }
}

void OpNode::ConcatBlobDesc(const std::vector<BlobDesc>& blob_descs,
                            const SbpParallel& sbp_parallel,
                            BlobDesc* concatenated_blob_desc) const {
  CHECK_EQ(blob_descs.size(), parallel_desc().parallel_num());
  if (sbp_parallel.has_split_parallel()) {
    int32_t axis = sbp_parallel.split_parallel().axis();
    // concat BlobDesc
    CHECK_GE(axis, 0);
    CHECK_LT(axis, blob_descs.at(0).shape().NumAxes());
    int64_t logical_blob_axis_dim = 0;
    for (const BlobDesc& blob_desc : blob_descs) {
      logical_blob_axis_dim += blob_desc.shape().At(axis);
    }
    CHECK_GE(logical_blob_axis_dim, parallel_desc().parallel_num());
    BalancedSplitter bs(logical_blob_axis_dim, parallel_desc().parallel_num());
    std::vector<BlobDesc> same_blob_descs(blob_descs);
    FOR_RANGE(int64_t, axis_parallel_id, 0, parallel_desc().parallel_num()) {
      CHECK_EQ(bs.At(axis_parallel_id).size(), blob_descs.at(axis_parallel_id).shape().At(axis));
      same_blob_descs.at(axis_parallel_id).mut_shape().Set(axis, logical_blob_axis_dim);
    }
    for (const BlobDesc& blob_desc : same_blob_descs) { CHECK(blob_desc == same_blob_descs.at(0)); }
    *concatenated_blob_desc = same_blob_descs.at(0);
  } else {
    // select first BlobDesc
    for (const BlobDesc& blob_desc : blob_descs) { CHECK(blob_desc == blob_descs.at(0)); }
    *concatenated_blob_desc = blob_descs.at(0);
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
    const BlobDesc& logical_blob_desc = ProducerOpNode4BnInOp(bn)->LogicalBlobDesc4Lbi(lbi);
    const SbpParallel& sbp_parallel = SbpParallel4Lbi(lbi);
    ForEachParallelBlobDesc(logical_blob_desc, sbp_parallel, [&](const BlobDesc& blob_desc) {
      lbi2parallel_id2blob_desc_[lbi].push_back(blob_desc);
    });
    CHECK_EQ(lbi2parallel_id2blob_desc_.at(lbi).size(), parallel_desc().parallel_num());
  }
}

void OpNode::ConcatLogicalOutputBlobDesc() {
  for (const std::string& bn : op().output_bns()) {
    const LogicalBlobId& lbi = op().BnInOp2Lbi(bn);
    const SbpParallel& sbp_parallel = SbpParallel4Lbi(lbi);
    ConcatBlobDesc(lbi2parallel_id2blob_desc_.at(lbi), sbp_parallel, MutLogicalBlobDesc4Lbi(lbi));
  }
}

void OpNode::CheckBlobDescs(const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  int64_t parallel_id = parallel_ctx->parallel_id();
  auto Check = [&](const std::string& bn) {
    const LogicalBlobId& lbi = op().BnInOp2Lbi(bn);
    if (lbi2parallel_id2blob_desc_.find(lbi) == lbi2parallel_id2blob_desc_.end()) { return; }
    CHECK_EQ(parallel_ctx->parallel_num(), lbi2parallel_id2blob_desc_.at(lbi).size());
    const BlobDesc& blob_desc_from_exec_graph = *GetBlobDesc4BnInOp(bn);
    const BlobDesc& blob_desc_from_op_graph = lbi2parallel_id2blob_desc_.at(lbi).at(parallel_id);
    CHECK_EQ(blob_desc_from_exec_graph.shape(), blob_desc_from_op_graph.shape());
    CHECK_EQ(blob_desc_from_exec_graph.data_type(), blob_desc_from_op_graph.data_type());
  };
  for (const std::string& bn : op().data_tmp_bns()) { Check(bn); }
  for (const std::string& bn : op().fw_buf_bns()) { Check(bn); }
  for (const std::string& bn : op().input_bns()) { Check(bn); }
  for (const std::string& bn : op().output_bns()) { Check(bn); }
  for (const std::string& bn : op().model_bns()) { Check(bn); }
  for (const std::string& bn : op().const_model_bns()) { Check(bn); }
  for (const std::string& bn : op().const_buf_bns()) { Check(bn); }
  for (const std::string& bn : op().forward_model_bns()) { Check(bn); }
}

void OpGraph::InferOpModelSize(HashMap<std::string, size_t>* op_name2model_size) {
  ForEachNode([&](OpNode* op_node) {
    size_t model_size = 0;
    for (const std::string& model_bn : op_node->op().model_bns()) {
      int64_t elem_cnt = op_node->NoParallelBlobDesc4BnInOp(model_bn)->shape().elem_cnt();
      model_size += elem_cnt * GetSizeOfDataType(job_desc_->DefaultDataType());
      model_size = RoundUp(model_size, kCudaAlignSize);
    }
    size_t parallel_num = op_node->parallel_desc().parallel_num();
    if (op_node->parallel_desc().policy() == ParallelPolicy::kModelParallel) {
      model_size = (model_size + parallel_num - 1) / parallel_num;
    }
    CHECK(op_name2model_size->emplace(op_node->op().op_name(), model_size).second);
  });
}

void OpGraph::Init() {
  InitNodes();
  ForEachNode(
      [&](OpNode* node) { CHECK(op_name2op_node_.emplace(node->op().op_name(), node).second); });
  InitEdges();
  FixOpParallelDesc();
  UpdateOpNodeHasInDiff();
  InferTimeShape();
  InferNoParallelBlobDesc();
  InferIsModelBlob();
  InferSbpParallel();
  InferLogicalBlobDesc();
}

void OpGraph::InitNodes() {
  auto ParallelConf4OpName = MakeGetterParallelConf4OpName(job_desc_->placement());
  for (const auto& op_conf : job_desc_->dlnet_conf().op()) {
    OpNode* node = new OpNode(ParallelDesc(*ParallelConf4OpName(op_conf.name())), op_conf);
    AddAllocatedNode(node);
  }
}

void OpGraph::InitEdges() {
  HashMap<LogicalBlobId, OpNode*> lbi2producer;
  ForEachNode([&](OpNode* op_node) {
    for (const auto& obn : op_node->op().output_bns()) {
      CHECK(lbi2producer.emplace(op_node->op().BnInOp2Lbi(obn), op_node).second);
    }
  });
  ForEachNode([&](OpNode* op_node) {
    HashMap<std::string, std::vector<LogicalBlobId>> producer_name2lbis;
    HashMap<std::string, HashMap<LogicalBlobId, std::vector<std::string>>>
        consumer_op_name2lbi2ibns;
    for (const auto& ibn : op_node->op().input_bns()) {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
      producer_name2lbis[lbi.op_name()].push_back(lbi);
      consumer_op_name2lbi2ibns[op_node->op().op_name()][lbi].push_back(ibn);
    }
    for (const auto& pair : producer_name2lbis) {
      const auto& lbis = pair.second;
      const auto& lbi2ibns = consumer_op_name2lbi2ibns.at(op_node->op().op_name());
      OpNode* producer = lbi2producer.at(lbis.at(0));
      Connect(producer, NewEdge(lbis, lbi2ibns), op_node);
    }
  });
}

void OpGraph::FixOpParallelDesc() const {
  ForEachNode([&](OpNode* node) { node->op().FixParallelDesc(node->mut_parallel_desc()); });
}

void OpGraph::UpdateOpNodeHasInDiff() const {
  TopoForEachNode([&](OpNode* op_node) {
    bool has_diff = false;
    for (OpEdge* edge : op_node->in_edges()) {
      if (edge->src_node()->has_in_diff() || edge->src_node()->has_model_diff()) {
        edge->set_has_diff(true);
        has_diff = true;
        break;
      }
    }
    op_node->set_has_in_diff(has_diff);
  });
}

void OpGraph::InferTimeShape() const {
  TopoForEachNode([&](OpNode* op_node) {
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_id(0);
    parallel_ctx.set_parallel_num(op_node->parallel_desc().parallel_num());
    parallel_ctx.set_policy(op_node->parallel_desc().policy());
    auto GetInputBlobTimeShape = [&](const std::string& bn_in_op) {
      return op_node->GetInputBlobTimeShape(bn_in_op);
    };
    op_node->op().InferOutputBlobTimeShapeIf(GetInputBlobTimeShape, &parallel_ctx,
                                             op_node->mut_out_blob_time_shape());
  });
}

int64_t OpGraph::CalcUnparalleledPieceSize() const {
  int64_t time_shape_dim0 = 0;
  int64_t max_time_shape_elem_cnt = 0;
  ForEachNode([&](OpNode* op_node) {
    const auto& out_blob_time_shape = op_node->out_blob_time_shape();
    if (time_shape_dim0 != 0) { CHECK_EQ(time_shape_dim0, out_blob_time_shape.At(0)); }
    time_shape_dim0 = out_blob_time_shape.At(0);
    max_time_shape_elem_cnt = std::max(max_time_shape_elem_cnt, out_blob_time_shape.elem_cnt());
  });
  CHECK_EQ(max_time_shape_elem_cnt % time_shape_dim0, 0);
  return max_time_shape_elem_cnt / time_shape_dim0;
}

void OpGraph::InferNoParallelBlobDesc() const {
  int64_t unparalleled_piece_size = CalcUnparalleledPieceSize();
  TopoForEachNode([&](OpNode* op_node) {
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_id(0);
    parallel_ctx.set_parallel_num(1);
    parallel_ctx.set_policy(op_node->parallel_desc().policy());
    // the real important data we want to get is:
    // a) model blobs' byte size;
    // b) number of axes of blobs' body shape;
    // Hence the argument record_piece_size can be any positive number, here it's 1
    op_node->op().InferBlobDescsIf(
        std::bind(&OpNode::NoParallelBlobDesc4BnInOp, op_node, std::placeholders::_1),
        &parallel_ctx, unparalleled_piece_size, [](OpContext*) {});
  });
}

void OpGraph::InferIsModelBlob() const {
  TopoForEachNode([&](OpNode* op_node) {
    op_node->op().InferIsModelBlob4OutputBlobsIf([&](const std::string& bn) -> bool* {
      return op_node->ProducerOpNode4BnInOp(bn)->MutIsModelBlob4Lbi(op_node->op().BnInOp2Lbi(bn));
    });
  });
}

void OpGraph::InferSbpParallel() const {
  TopoForEachNode([&](OpNode* op_node) {
    HashMap<std::string, SbpInferHint> ibn2sbp_infer_hint;
    for (const std::string& ibn : op_node->op().input_bns()) {
      const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
      OpNode* producer = op_node->SrcNode4InputBnInOp(ibn);
      bool is_model_blob = producer->IsModelBlob4Lbi(lbi);
      const ParallelDesc& parallel_desc = op_node->parallel_desc();
      int64_t num_axes = producer->NoParallelBlobDesc4Lbi(lbi).shape().NumAxes();
      const auto& sbp = producer->SbpParallel4Lbi(lbi);
      ibn2sbp_infer_hint.emplace(ibn, SbpInferHint(is_model_blob, parallel_desc, num_axes, sbp));
    }
    auto SbpParallel4BnInOp = [&](const std::string& bn) -> SbpParallel* {
      return op_node->MutSbpParallel4Lbi(op_node->op().BnInOp2Lbi(bn));
    };
    auto SbpInferHint4Ibn = [&](const std::string& ibn) -> const SbpInferHint& {
      return ibn2sbp_infer_hint.at(ibn);
    };
    op_node->op().InferInputOutputSbpParallelIf(SbpParallel4BnInOp, SbpInferHint4Ibn,
                                                op_node->parallel_desc());
  });
}

void OpGraph::InferLogicalBlobDesc() const {
  TopoForEachNode([&](OpNode* op_node) {
    auto* lbi2parallel_id2blob_desc = op_node->mut_lbi2parallel_id2blob_desc();
    op_node->SplitLogicalInputBlobDesc();
    int64_t parallel_num = op_node->parallel_desc().parallel_num();
    const auto& input_bns = op_node->op().input_bns();
    FOR_RANGE(int64_t, parallel_id, 0, parallel_num) {
      auto BlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
        const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(bn);
        if (std::find(input_bns.begin(), input_bns.end(), bn) != input_bns.end()) {
          CHECK(lbi2parallel_id2blob_desc->find(lbi) != lbi2parallel_id2blob_desc->end());
          CHECK_EQ(lbi2parallel_id2blob_desc->at(lbi).size(), parallel_num);
        } else if (lbi2parallel_id2blob_desc->find(lbi) == lbi2parallel_id2blob_desc->end()) {
          (*lbi2parallel_id2blob_desc)[lbi].resize(parallel_num);
        } else {
          CHECK_EQ(lbi2parallel_id2blob_desc->at(lbi).size(), parallel_num);
        }
        return &(*lbi2parallel_id2blob_desc)[lbi][parallel_id];
      };
      ParallelContext parallel_ctx;
      parallel_ctx.set_parallel_id(parallel_id);
      parallel_ctx.set_parallel_num(parallel_num);
      parallel_ctx.set_policy(op_node->parallel_desc().policy());
      op_node->op().InferBlobDescsIf(BlobDesc4BnInOp, &parallel_ctx, job_desc_->RecordPieceSize(),
                                     [](OpContext*) {});
    }
    op_node->ConcatLogicalOutputBlobDesc();
  });
}

BalancedSplitter OpGraph::GetBalancedSplitter(const std::string& op_name,
                                              const LogicalBlobId& lbi) const {
  OpNode* op_node = op_name2op_node_.at(GetOpNameKey(op_name, lbi));
  const SbpParallel& sbp_parallel = GetSbpParallel(op_name, lbi);
  CHECK(sbp_parallel.has_split_parallel());
  int64_t split_num = GetSplitNum(op_name, lbi);
  if (IsDataBlob(op_name, lbi)) {
    CHECK_EQ(split_num % op_node->parallel_desc().parallel_num(), 0);
  } else {
    CHECK(IsModelBlob(op_name, lbi));
    CHECK_GE(split_num, op_node->parallel_desc().parallel_num());
  }
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
  return op_node->ProducerOpNode4Lbi(lbi)->LogicalBlobDesc4Lbi(lbi_key).shape().At(
      sbp_parallel.split_parallel().axis());
}

const SbpParallel& OpGraph::GetSbpParallel(const std::string& op_name,
                                           const LogicalBlobId& lbi) const {
  return op_name2op_node_.at(GetOpNameKey(op_name, lbi))
      ->SbpParallel4Lbi(GetLogicalBlobIdKey(op_name, lbi));
}

DataType OpGraph::GetBlobDataType(const LogicalBlobId& lbi) const {
  return op_name2op_node_.at(lbi.op_name())
      ->NoParallelBlobDesc4Lbi(GetLogicalBlobIdKey(lbi.op_name(), lbi))
      .data_type();
}

bool OpGraph::IsModelBlob(const std::string& op_name, const LogicalBlobId& lbi) const {
  return op_name2op_node_.at(GetOpNameKey(op_name, lbi))
      ->IsModelBlob4Lbi(GetLogicalBlobIdKey(op_name, lbi));
}

bool OpGraph::IsDataBlob(const std::string& op_name, const LogicalBlobId& lbi) const {
  return !IsModelBlob(op_name, lbi);
}

void OpGraph::CheckBlobDescs(const std::string& op_name,
                             const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  if (op_name2op_node_.find(op_name) == op_name2op_node_.end()) { return; }
  op_name2op_node_.at(op_name)->CheckBlobDescs(GetBlobDesc4BnInOp, parallel_ctx);
}

void OpGraph::ForEachPseudoChain(
    const std::function<void(const HashSet<OpNode*>&)>& Handler) const {
  auto IsReachable = MakePredicatorIsReachable();
  ForEachComponentWithSameDataParallelDescAndTimeShape(
      [&](const std::vector<OpNode*>& nodes) { ForEachPseudoChain(nodes, IsReachable, Handler); });
}

std::function<bool(OpNode* src, OpNode* dst)> OpGraph::MakePredicatorIsReachable() const {
  auto node2ancestors_ptr = std::make_shared<HashMap<OpNode*, HashSet<OpNode*>>>();
  TopoForEachNode([&](OpNode* node) {
    node->ForEachNodeOnInEdge([&](OpNode* in_node) {
      (*node2ancestors_ptr)[node].insert(in_node);
      (*node2ancestors_ptr)[node].insert((*node2ancestors_ptr)[in_node].begin(),
                                         (*node2ancestors_ptr)[in_node].end());
    });
  });
  return [node2ancestors_ptr](OpNode* src, OpNode* dst) -> bool {
    return node2ancestors_ptr->at(dst).find(src) != node2ancestors_ptr->at(dst).end();
  };
}

void OpGraph::ForEachComponentWithSameDataParallelDescAndTimeShape(
    const std::function<void(const std::vector<OpNode*>&)>& Handler) const {
  auto WithSameDataParallelDescAndTimeShape = [](OpNode* src, OpNode* dst) -> bool {
    if (src->parallel_desc().policy() != ParallelPolicy::kDataParallel) { return false; }
    if (dst->parallel_desc().policy() != ParallelPolicy::kDataParallel) { return false; }
    if (src->in_edges().empty()) { return false; }
    if (*src->GetInputBlobTimeShape() != src->out_blob_time_shape()) { return false; }
    if (*dst->GetInputBlobTimeShape() != dst->out_blob_time_shape()) { return false; }
    return src->parallel_desc() == dst->parallel_desc()
           && src->out_blob_time_shape() == dst->out_blob_time_shape();
  };
  auto ForEachNext = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    node->ForEachNodeOnInEdge([&](OpNode* in_node) {
      if (WithSameDataParallelDescAndTimeShape(in_node, node)) { Handler(in_node); }
    });
    node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
      if (WithSameDataParallelDescAndTimeShape(node, out_node)) { Handler(out_node); }
    });
  };
  HashMap<OpNode*, int32_t> op_node2component_id;
  int32_t cur_component_id = 0;
  ForEachNode([&](OpNode* start) {
    if (op_node2component_id.find(start) != op_node2component_id.end()) { return; }
    ++cur_component_id;
    BfsForEachNode({start}, ForEachNext, [&](OpNode* node) {
      CHECK(op_node2component_id.emplace(node, cur_component_id).second);
    });
  });
  HashMap<int32_t, std::vector<OpNode*>> component_id2op_nodes;
  for (const auto& pair : op_node2component_id) {
    component_id2op_nodes[pair.second].push_back(pair.first);
  }
  for (const auto& pair : component_id2op_nodes) { Handler(pair.second); }
}

void OpGraph::ForEachPseudoChain(
    const std::vector<OpNode*>& nodes,
    const std::function<bool(OpNode* src, OpNode* dst)>& IsReachable,
    const std::function<void(const HashSet<OpNode*>&)>& Handler) const {
  if (nodes.size() <= 1) { return; }
  if (nodes.front()->parallel_desc().device_type() == DeviceType::kCPU) { return; }
  if (nodes.front()->parallel_desc().policy() != ParallelPolicy::kDataParallel) { return; }
  HashSet<OpNode*> all_nodes(nodes.begin(), nodes.end());
  while (all_nodes.size() > 1) {
    HashSet<OpNode*> chain;
    ReverseTopoGetPseudoChain(all_nodes, &chain, IsReachable);
    Handler(chain);
    for (OpNode* node_in_chain : chain) { all_nodes.erase(node_in_chain); }
  }
}

void OpGraph::ReverseTopoGetPseudoChain(
    const HashSet<OpNode*>& op_nodes, HashSet<OpNode*>* pseudo_chain_nodes,
    const std::function<bool(OpNode* src, OpNode* dst)>& IsReachable) const {
  // get sink nodes
  std::list<OpNode*> sinks;
  auto IsSink = [&](OpNode* node) {
    for (OpNode* inner_node : op_nodes) {
      if (IsReachable(node, inner_node)) { return false; }
    }
    return true;
  };
  for (OpNode* op_node : op_nodes) {
    if (IsSink(op_node)) { sinks.push_back(op_node); }
  }
  // generate connections of subgraph
  HashMap<OpNode*, std::vector<OpNode*>> node2in_nodes;
  HashMap<OpNode*, std::vector<OpNode*>> node2out_nodes;
  auto IsInSubset = [&](OpNode* node) { return op_nodes.find(node) != op_nodes.end(); };
  auto ReachableToAnySink = [&](OpNode* node) {
    for (OpNode* sink : sinks) {
      if (node == sink) { return true; }
      if (IsReachable(node, sink)) { return true; }
    }
    return false;
  };
  auto AnyOutputNodesNotInSubsetAndReachableIntoSink = [&](OpNode* node) {
    for (OpEdge* edge : node->out_edges()) {
      if (!IsInSubset(edge->dst_node()) && ReachableToAnySink(edge->dst_node())) { return true; }
    }
    return false;
  };
  for (OpNode* node : op_nodes) {
    if (AnyOutputNodesNotInSubsetAndReachableIntoSink(node)) { continue; }
    node->ForEachNodeOnOutEdge([&](OpNode* out_node) {
      if (IsInSubset(out_node)) {
        node2in_nodes[out_node].push_back(node);
        node2out_nodes[node].push_back(out_node);
      }
    });
  }
  // get chain nodes
  auto ForEachInNode = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    for (OpNode* in_node : node2in_nodes[node]) { Handler(in_node); }
  };
  auto ForEachOutNode = [&](OpNode* node, const std::function<void(OpNode*)>& Handler) {
    for (OpNode* out_node : node2out_nodes[node]) { Handler(out_node); }
  };
  TopoForEachNode(sinks, ForEachOutNode, ForEachInNode,
                  [&](OpNode* node) { CHECK(pseudo_chain_nodes->emplace(node).second); });
}

std::string OpGraph::GetOpNameKey(const std::string& op_name, const LogicalBlobId& lbi) const {
  CHECK(!lbi.has_is_packed_id());
  std::string op_name_key;
  if (op_name2op_node_.find(op_name) == op_name2op_node_.end()) {
    CHECK(lbi.has_clone_id());
    return lbi.op_name();
  } else {
    CHECK(!lbi.has_clone_id());
    return op_name;
  }
}

LogicalBlobId OpGraph::GetLogicalBlobIdKey(const std::string& op_name,
                                           const LogicalBlobId& lbi) const {
  CHECK(!lbi.has_is_packed_id());
  if (op_name2op_node_.find(op_name) == op_name2op_node_.end()) {
    CHECK(lbi.has_clone_id());
    LogicalBlobId lbi_key;
    lbi_key.set_op_name(lbi.op_name());
    lbi_key.set_blob_name(lbi.blob_name());
    return lbi_key;
  } else {
    CHECK(!lbi.has_clone_id());
    return lbi;
  }
}

}  // namespace oneflow
