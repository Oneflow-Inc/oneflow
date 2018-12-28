#include "oneflow/core/operator/pack_op.h"

namespace oneflow {

void PackOp::InitFromOpConf() {
  CHECK(op_conf().has_pack_conf());

  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

void PackOp::InferOutBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  std::vector<int64_t> dim_vec(GetTimeShape4BnInOp("in")->dim_vec());
  int32_t pack_num = GetPackNum(parallel_ctx->parallel_num());
  CHECK_EQ(pack_num, dim_vec.back());
  dim_vec.pop_back();
  *time_shape = Shape(dim_vec);
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
    return 0;
  }
}

REGISTER_OP(OperatorConf::kPackConf, PackOp);

}  // namespace oneflow
