#include "operator/copy_op.h"
#include "operator/operator_manager.h"

namespace oneflow {

void CopyOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_copy_conf());
  mut_op_conf() = op_conf;
  for (int64_t i = 0; i < op_conf.copy_conf().copied_lbns_size(); ++i) {
    std::string ibn = "in_" + std::to_string(i);
    EnrollInputBn(ibn);
    CHECK(ibn2lbn_.emplace(ibn, op_conf.copy_conf().copied_lbns(i)).second);
    std::string obn = "out_" + std::to_string(i);
    EnrollOutputBn(obn);
    CHECK(obn2lbn_.emplace(obn, op_conf.copy_conf().copied_lbns(i)).second);
  }
}
std::string CopyOp::GetValueFromPbOpConf(const std::string& k) const {
  return GetValueFromPbMessage(op_conf().copy_conf(), k);
}
void CopyOp::InitFromOperatorProto(const OperatorProto& operatorproto) {
  CHECK(operatorproto.has_copy_op());
  Operator::InitFromOperatorProto(operatorproto);
  ibn2lbn_ = PbMap2HashMap(operatorproto.copy_op().ibn2lbn());
  obn2lbn_ = PbMap2HashMap(operatorproto.copy_op().obn2lbn());
}

OperatorProto CopyOp::ToOperatorProto() {
  OperatorProto operatorproto = Operator::ToOperatorProto();
  CopyOpProto copyopproto;
  *(copyopproto.mutable_ibn2lbn()) = HashMap2PbMap(ibn2lbn_);
  *(copyopproto.mutable_obn2lbn()) = HashMap2PbMap(obn2lbn_);
  *(operatorproto.mutable_copy_op()) = copyopproto;
  return operatorproto;
}

void CopyOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy,
    uint64_t parallel_id,
    uint64_t parallel_size) const {
  CHECK_EQ(output_bns().size(), input_bns().size());
  for(size_t i = 0;i < output_bns().size();++ i){
    std::string obn = output_bns().at(i);
    std::string ibn = input_bns().at(i);
    *GetShapePtr4BnInOp(obn) = *GetShapePtr4BnInOp(ibn);
  }
}

REGISTER_OP(OperatorConf::kCopyConf, CopyOp);

} // namespace oneflow
