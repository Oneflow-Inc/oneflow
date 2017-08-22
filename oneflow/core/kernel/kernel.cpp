#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

void Kernel::InitFromOpProto(const OperatorProto& op_proto) {
  Operator* op = CreateOp(op_proto.op_conf().op_type_case());
  op->InitFromProto(op_proto);
  op_.reset(op);
}

void Kernel::InitModelBlobs(
    const KernelCtx& ctx, ParallelPolicy policy, int64_t parallel_id,
    int64_t parallel_num, const Snapshot* snapshot,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (snapshot) {
    int32_t part_id = -1;
    int32_t part_num = -1;
    if (policy == kDataParallel) {
      part_id = 0;
      part_num = 1;
      CHECK_EQ(parallel_id, 0);
    } else if (policy == kModelParallel) {
      part_id = parallel_id;
      part_num = parallel_num;
    } else {
      UNEXPECTED_RUN();
    }
    InitModelBlobsWithSnapshot(ctx, part_id, part_num, snapshot, BnInOp2Blob);
  } else {
    uint32_t random_seed = reinterpret_cast<uint64_t>(ctx.other);
    std::mt19937 random_seed_gen(random_seed);
    InitModelBlobsWithRandomSeed(ctx, random_seed_gen, BnInOp2Blob);
  }
}

}  // namespace oneflow
