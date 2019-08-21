#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_nccl_tuple_broadcast_conf());
  const NcclTupleBroadcastOpConf& conf = op.op_conf().nccl_tuple_broadcast_conf();
  FOR_RANGE(int64_t, i, 0, conf.in_size()) {
    const std::string ibn = GenRepeatedBn("in", i);
    if (DiffLbi4BnInOp(ibn) != nullptr) {
      OperatorConf nccl_tuple_reduce_op{};
      nccl_tuple_reduce_op.set_name("System-AutoGrad-" + op.op_name());
      NcclTupleReduceOpConf* tuple_reduce_conf =
          nccl_tuple_reduce_op.mutable_nccl_tuple_reduce_conf();
      const std::string obn = GenRepeatedBn("out", i);
      *tuple_reduce_conf->mutable_in()->Add() = GenLogicalBlobName(*DiffLbi4BnInOp(obn));
      *tuple_reduce_conf->mutable_out()->Add() = "out";
      *tuple_reduce_conf->mutable_root()->Add() = conf.root(i);
      op_confs->push_back(nccl_tuple_reduce_op);
      DiffLbi4BnInOp(ibn)->set_op_name(nccl_tuple_reduce_op.name());
      DiffLbi4BnInOp(ibn)->set_blob_name("out");
    }
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kNcclTupleBroadcastConf, &GenerateBackwardOpConf);

}  // namespace oneflow
