#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

void Kernel::InitFromOpProto(const OperatorProto& op_proto) { TODO(); }

void Kernel::InitModelBlobs(
    const KernelCtx& ctx, ParallelPolicy policy, int64_t parallel_id,
    int64_t parallel_num, const Snapshot* snapshot,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int32_t part_id = -1;
  int32_t part_num = -1;
  if (policy == kDataParallel) {
    part_id = 0;
    part_num = 1;
  } else if (policy == kModelParallel) {
    part_id = parallel_id;
    part_num = parallel_num;
  } else {
    UNEXPECTED_RUN();
  }
  std::string model_load_dir = op()->op_conf().model_load_dir();
  if (model_load_dir == "" && snapshot) {
    model_load_dir = snapshot->GetDirFromOpName(op()->op_name());
  }
  if (model_load_dir == "") {
    uint32_t random_seed = reinterpret_cast<uint64_t>(ctx.other);
    std::mt19937 random_seed_gen(random_seed);
    InitModelBlobsWithRandomSeed(ctx, random_seed_gen, BnInOp2Blob);
  } else {
    InitModelBlobsWithDir(ctx, part_id, part_num, model_load_dir, BnInOp2Blob);
  }
}

}  // namespace oneflow
