#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

void Kernel::Init(const ParallelContext* parallel_ctx,
                  const KernelConf& kernel_conf) {
  kernel_conf_ = kernel_conf;
  VirtualKernelInit(parallel_ctx);
}

void Kernel::InitModelBlobs(
    const KernelCtx& ctx, const ParallelContext* parallel_ctx,
    const Snapshot* snapshot,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::string model_load_dir = kernel_conf().op_conf().model_load_dir();
  if (model_load_dir == "" && snapshot) {
    model_load_dir = snapshot->GetDirFromOpName(op_conf().name());
  }
  if (model_load_dir == "") {
    uint32_t random_seed = reinterpret_cast<uint64_t>(ctx.other);
    std::mt19937 random_seed_gen(random_seed);
    InitModelBlobsWithRandomSeed(ctx, random_seed_gen, BnInOp2Blob);
  } else {
    int32_t part_id;
    int32_t part_num;
    std::tie(part_id, part_num) =
        GetPartIdAndPartNumFromParallelCtx(parallel_ctx);
    InitModelBlobsWithDir(ctx, part_id, part_num, model_load_dir, BnInOp2Blob);
  }
}

void Kernel::InitModelTmpBlobs(
    const KernelCtx& ctx, const ParallelContext* parallel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNEXPECTED_RUN();
}

void Kernel::Launch(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (kernel_conf_.is_forward()) {
    Forward(ctx, BnInOp2Blob);
  } else {
    Backward(ctx, BnInOp2Blob);
  }
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

std::unique_ptr<const Kernel> ConstructKernel(
    DeviceType device_type, const ParallelContext* parallel_ctx,
    const KernelConf& conf) {
  OperatorConf::OpTypeCase opcase = conf.op_conf().op_type_case();
  Kernel* rptr = GetCreatorsMap().at(opcase)(device_type, conf);
  rptr->Init(parallel_ctx, conf);
  return std::unique_ptr<const Kernel>(rptr);
}

template class KernelIf<DeviceType::kCPU>;
template class KernelIf<DeviceType::kGPU>;

}  // namespace oneflow
