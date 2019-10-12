#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_nccl_tuple_broadcast_conf());
  const NcclTupleBroadcastOpConf& conf = op.op_conf().nccl_tuple_broadcast_conf();
  OperatorConf nccl_tuple_reduce_op{};
  nccl_tuple_reduce_op.set_name("System-AutoGrad-" + op.op_name());
  NcclTupleReduceOpConf* tuple_reduce_conf = nccl_tuple_reduce_op.mutable_nccl_tuple_reduce_conf();
  OperatorConf tuple_identity_op{};
  tuple_identity_op.set_name("System-AutoGrad-" + op.op_name() + "-TupleIdentity");
  TupleIdentityOpConf* tuple_identity_conf = tuple_identity_op.mutable_tuple_identity_conf();
  tuple_reduce_conf->set_nccl_order_hint(-conf.nccl_order_hint());
  FOR_RANGE(int64_t, i, 0, conf.in_size()) {
    const std::string ibn = GenRepeatedBn("in", i);
    LogicalBlobId* diff_lbi = DiffLbi4BnInOp(ibn);
    if (diff_lbi != nullptr) {
      const std::string obn = GenRepeatedBn("out", i);
      *tuple_identity_conf->mutable_in()->Add() = GenLogicalBlobName(*DiffLbi4BnInOp(obn));
      *tuple_identity_conf->mutable_out()->Add() = obn;
      *tuple_reduce_conf->mutable_in()->Add() = tuple_identity_op.name() + "/" + obn;
      *tuple_reduce_conf->mutable_out()->Add() = obn;
      *tuple_reduce_conf->mutable_root()->Add() = conf.root(i);
      diff_lbi->set_op_name(nccl_tuple_reduce_op.name());
      diff_lbi->set_blob_name(obn);
    }
  }
  if (tuple_reduce_conf->in_size() > 0) {
    op_confs->push_back(nccl_tuple_reduce_op);
    op_confs->push_back(tuple_identity_op);
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kNcclTupleBroadcastConf, &GenerateBackwardOpConf);

}  // namespace oneflow
