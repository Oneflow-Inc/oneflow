#include "oneflow/core/operator/add_op.h"

namespace oneflow {

void AddOp::VirtualInitFromOpConf() { CHECK(op_conf().has_add_conf()); }
/*
void AddOp::VirtualInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {

  int64_t mem_shared_id = std::stol(NewUniqueId());
  // std::vector<std::string> idbns = input_diff_bns();

  LOG(INFO) << "op_name:" << op_name();
  // LOG(INFO) << "add_op:" << idbns.size();
  for (size_t i = 0; i < input_diff_bns().size(); ++i) {
    std::string idbn = input_diff_bns().Get(i);
    LOG(INFO) << idbn;
    BlobDesc* in_diff = GetBlobDesc4BnInOp(idbn);
    in_diff->set_mem_shared_id(mem_shared_id);
    // GetBlobDesc4BnInOp(input_diff_bns().Get(i))->set_mem_shared_id(mem_shared_id);
  }
}
*/
const PbMessage& AddOp::GetCustomizedConf() const { return op_conf().add_conf(); }

REGISTER_OP(OperatorConf::kAddConf, AddOp);

}  // namespace oneflow
