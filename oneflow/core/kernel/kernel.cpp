#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

void Kernel::InitFromOpProto(const OperatorProto& op_proto) {
  Operator* op = CreateOp(op_proto.op_conf().op_type_case());
  op->InitFromProto(op_proto);
  op_.reset(op);
}

void Kernel::InitModelAndModelTmpBlobs(
    const KernelCtx& ctx, ParallelPolicy policy, int64_t parallel_id,
    int64_t parallel_num, const Snapshot* snapshot,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (snapshot) {
    InitModelAndModelTmpBlobsWithSnapshot(ctx, policy, parallel_id,
                                          parallel_num, snapshot, BnInOp2Blob);
  } else {
    uint32_t random_seed = reinterpret_cast<uint64_t>(ctx.other);
    std::mt19937 random_seed_gen(random_seed);
    InitModelAndModelTmpBlobsWithRandomSeed(ctx, random_seed_gen, BnInOp2Blob);
  }
}

}  // namespace oneflow
