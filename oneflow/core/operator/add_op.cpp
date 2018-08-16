#include "oneflow/core/operator/add_op.h"

namespace oneflow {

void AddOp::VirtualInitFromOpConf() { CHECK(op_conf().has_add_conf()); }
const PbMessage& AddOp::GetCustomizedConf() const { return op_conf().add_conf(); }
void AddOp::FixInDiffBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  if (!Global<JobDesc>::Get()->enable_blob_mem_sharing()) { return; }
  int64_t mem_shared_id = oneflow_cast<int64_t>(NewUniqueId());
  FOR_RANGE(size_t, i, 0, input_diff_bns().size()) {
    GetBlobDesc4BnInOp(input_diff_bns().Get(i))->set_mem_shared_id(mem_shared_id);
  }
}

REGISTER_OP(OperatorConf::kAddConf, AddOp);

}  // namespace oneflow
