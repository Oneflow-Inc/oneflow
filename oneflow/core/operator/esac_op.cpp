#include "oneflow/core/operator/esac_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void EsacOp::InitFromOpConf() {
  EnrollRepeatedInputBn("in", false);
  EnrollOutputBn("out", false);
}

Maybe<void> EsacOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->mut_shape() = Shape({1});
  const DataType data_type = op_conf().esac_conf().data_type();
  CHECK_OR_RETURN(IsIntegralDataType(data_type));
  out->set_data_type(data_type);
  return Maybe<void>::Ok();
}

const PbMessage& EsacOp::GetCustomizedConf() const { return op_conf().esac_conf(); }

Maybe<void> EsacOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("out")->clear_value();
  return Maybe<void>::Ok();
}

void EsacOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {}

LogicalNode* EsacOp::NewProperLogicalNode() const { return new EsacLogicalNode(); }

REGISTER_CPU_OP(OperatorConf::kEsacConf, EsacOp);

}  // namespace oneflow
