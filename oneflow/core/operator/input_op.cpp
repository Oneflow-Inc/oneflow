#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/operator/input_op.h"
#include "oneflow/core/operator/interface_op_util.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

void CheckOpConf(const OperatorConf& op_conf) {
  if (op_conf.input_conf().has_blob_conf()) {
    if (op_conf.input_conf().blob_conf().has_dim1_valid_num()) { TODO(); }
    if (op_conf.input_conf().blob_conf().has_dim2_valid_num()) { TODO(); }
  }
}

}  // namespace

void InputOp::InitFromOpConf() {
  CHECK(op_conf().has_input_conf());
  if (op_conf().input_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", false);
}

const PbMessage& InputOp::GetCustomizedConf() const { return op_conf().input_conf(); }

void InputOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, int64_t record_piece_size) const {
  CheckOpConf(op_conf());
  InterfaceOpUtil::InferOutBlobDesc(op_conf().input_conf().blob_conf(), GetBlobDesc4BnInOp("out"),
                                    parallel_ctx, record_piece_size);
}

void InputOp::InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = op_conf().input_conf().blob_conf().has_batch_dim();
}

void InputOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  InterfaceOpUtil::GetInputLikeOpSbpSignature(op_conf().input_conf().blob_conf(), input_bns(),
                                              output_bns(),
                                              sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kInputConf, InputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_MEM_BLOCK_NUM(OperatorConf::kInputConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kInputConf);

}  // namespace oneflow
