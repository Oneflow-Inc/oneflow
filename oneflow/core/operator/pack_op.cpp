#include "oneflow/core/operator/pack_op.h"

namespace oneflow {

void PackOp::InitFromOpConf() {
  CHECK(op_conf().has_pack_conf());

  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

int32_t PackOp::GetPackNum(int64_t parallel_num) const {
  CHECK(op_conf().has_pack_conf());
  const PackOpConf& conf = op_conf().pack_conf();
  if (conf.has_pack_num()) {
    return conf.pack_num();
  } else if (conf.has_pack_num_per_record()) {
    CHECK_EQ(Global<JobDesc>::Get()->PieceSize() % parallel_num, 0);
    int64_t pack_num =
        Global<JobDesc>::Get()->PieceSize() / parallel_num * conf.pack_num_per_record();
    CHECK_LE(pack_num, static_cast<int64_t>(GetMaxVal<int32_t>()));
    return static_cast<int32_t>(pack_num);
  } else {
    UNIMPLEMENTED();
  }
}

REGISTER_OP(OperatorConf::kPackConf, PackOp);

}  // namespace oneflow
