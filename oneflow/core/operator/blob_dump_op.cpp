#include "oneflow/core/operator/blob_dump_op.h"

namespace oneflow {

void BlobDumpOp::InitFromOpConf() {
  CHECK(op_conf().has_blob_dump_conf());
  const BlobDumpOpConf& conf = op_conf().blob_dump_conf();

  FOR_RANGE(int32_t, i, 0, conf.in_size()) { EnrollInputBn("in_" + std::to_string(i), false); }
}

const PbMessage& BlobDumpOp::GetCustomizedConf() const { return op_conf().blob_dump_conf(); }

LogicalBlobId BlobDumpOp::Lbi4InputBn(const std::string& input_bn) const {
  CHECK_STREQ(input_bn.substr(0, 3).c_str(), "in_");
  return GenLogicalBlobId(
      op_conf().blob_dump_conf().in(oneflow_cast<int32_t>(input_bn.substr(3))).lbn());
}

LogicalBlobId BlobDumpOp::ibn2lbi(const std::string& input_bn) const {
  return Lbi4InputBn(input_bn);
}

REGISTER_OP(OperatorConf::kBlobDumpConf, BlobDumpOp);

}  // namespace oneflow
