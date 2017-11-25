#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

void Kernel::Init(const KernelConf& kernel_conf) {
  kernel_conf_ = kernel_conf;
  VirtualKernelInit();
}

void Kernel::InitModelBlobs(
    const KernelCtx& ctx, const ParallelContext& parallel_ctx,
    const Snapshot* snapshot,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int32_t part_id = -1;
  int32_t part_num = -1;
  if (parallel_ctx.policy() == kDataParallel) {
    part_id = 0;
    part_num = 1;
  } else if (parallel_ctx.policy() == kModelParallel) {
    part_id = parallel_ctx.parallel_id();
    part_num = parallel_ctx.parallel_num();
  } else {
    UNEXPECTED_RUN();
  }
  std::string model_load_dir = kernel_conf().op_conf().model_load_dir();
  if (model_load_dir == "" && snapshot) {
    model_load_dir = snapshot->GetDirFromOpName(op_conf().name());
  }
  if (model_load_dir == "") {
    uint32_t random_seed = reinterpret_cast<uint64_t>(ctx.other);
    std::mt19937 random_seed_gen(random_seed);
    InitModelBlobsWithRandomSeed(ctx, random_seed_gen, BnInOp2Blob);
  } else {
    InitModelBlobsWithDir(ctx, part_id, part_num, model_load_dir, BnInOp2Blob);
  }
}

void Kernel::InitModelTmpBlobs(
    const KernelCtx& ctx, const ParallelContext& parallel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNEXPECTED_RUN();
}

const std::string& Kernel::Lbn4BnInOp(const std::string& bn_in_op) const {
  return kernel_conf_.bn_in_op2lbn().at(bn_in_op);
}

void Kernel::InitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNEXPECTED_RUN();
}
void Kernel::InitModelBlobsWithDir(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNEXPECTED_RUN();
}

void Kernel::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardDataContent(ctx, BnInOp2Blob);
  if (kernel_conf_.need_do_data_id()) { ForwardDataId(ctx, BnInOp2Blob); }
}

void Kernel::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BackwardDataContent(ctx, BnInOp2Blob);
  if (kernel_conf_.need_do_data_id()) { BackwardDataId(ctx, BnInOp2Blob); }
}

template<DeviceType device_type>
void KernelIf<device_type>::ForwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (kernel_conf().input_bns().size() == 1) {
    const Blob* in_blob = BnInOp2Blob(kernel_conf().input_bns(0));
    CopyDataIdToAllOb(ctx.device_ctx, BnInOp2Blob, in_blob);
  } else {
    CHECK_EQ(kernel_conf().input_bns().size(),
             kernel_conf().output_bns().size());
    FOR_RANGE(size_t, i, 0, kernel_conf().input_bns().size()) {
      const std::string& ibn = kernel_conf().input_bns(i);
      const std::string& obn = kernel_conf().output_bns(i);
      Blob* in_blob = BnInOp2Blob(ibn);
      Blob* out_blob = BnInOp2Blob(obn);
      out_blob->CopyDataIdFrom<device_type>(ctx.device_ctx, in_blob);
    }
  }
}

template<DeviceType device_type>
void KernelIf<device_type>::BackwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<DeviceType device_type>
void KernelIf<device_type>::CopyDataIdToAllOb(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const Blob* blob) const {
  for (const std::string& obn : kernel_conf().output_bns()) {
    Blob* output_blob = BnInOp2Blob(obn);
    output_blob->CopyDataIdFrom<device_type>(ctx, blob);
  }
}

namespace {

HashMap<int, KernelCreator1>& GetCreatorsMap() {
  static HashMap<int, KernelCreator1> obj;
  return obj;
}

}  // namespace

void AddKernelCreator(OperatorConf::OpTypeCase opcase, KernelCreator1 creator) {
  CHECK(GetCreatorsMap().emplace(opcase, creator).second);
}
void AddKernelCreator(OperatorConf::OpTypeCase opcase, KernelCreator2 creator) {
  AddKernelCreator(opcase, [creator](DeviceType type, const KernelConf&) {
    return creator(type);
  });
}
void AddKernelCreator(OperatorConf::OpTypeCase opcase, KernelCreator3 creator) {
  AddKernelCreator(opcase, [creator](DeviceType, const KernelConf& conf) {
    return creator(conf);
  });
}
void AddKernelCreator(OperatorConf::OpTypeCase opcase, KernelCreator4 creator) {
  AddKernelCreator(
      opcase, [creator](DeviceType, const KernelConf&) { return creator(); });
}

std::unique_ptr<const Kernel> ConstructKernel(DeviceType device_type,
                                              const KernelConf& conf) {
  OperatorConf::OpTypeCase opcase = conf.op_conf().op_type_case();
  Kernel* rptr = GetCreatorsMap().at(opcase)(device_type, conf);
  rptr->Init(conf);
  return std::unique_ptr<const Kernel>(rptr);
}

template class KernelIf<DeviceType::kCPU>;
template class KernelIf<DeviceType::kGPU>;

}  // namespace oneflow
