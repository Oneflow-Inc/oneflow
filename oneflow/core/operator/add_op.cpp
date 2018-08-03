#include "oneflow/core/operator/add_op.h"

namespace oneflow {

void AddOp::VirtualInitFromOpConf() { CHECK(op_conf().has_add_conf()); }
void AddOp::RefineDiffBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  int64_t mem_shared_id = std::stol(NewUniqueId());

  for (size_t i = 0; i < input_diff_bns().size(); ++i) {
    GetBlobDesc4BnInOp(input_diff_bns().Get(i))->set_mem_shared_id(mem_shared_id);
  }
}
const PbMessage& AddOp::GetCustomizedConf() const { return op_conf().add_conf(); }

REGISTER_OP(OperatorConf::kAddConf, AddOp);

}  // namespace oneflow
