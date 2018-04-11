#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

void Kernel::Init(const ParallelContext* parallel_ctx,
                  const KernelConf& kernel_conf) {
  kernel_conf_ = kernel_conf;
  VirtualKernelInit(parallel_ctx);
}

void Kernel::InitModelAndModelTmp(
    const KernelCtx& ctx, const ParallelContext* parallel_ctx,
    const Snapshot* snapshot,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitPureModelTmpBlobs(ctx.device_ctx, BnInOp2Blob);
  std::string model_load_dir = kernel_conf().op_conf().model_load_dir();
  if (model_load_dir == "" && snapshot) {
    model_load_dir = snapshot->GetDirFromOpName(op_conf().name());
  }
  if (model_load_dir == "") {
    std::mt19937* random_seed_gen = static_cast<std::mt19937*>(ctx.other);
    InitModelBlobsWithRandomSeed(ctx.device_ctx, random_seed_gen, BnInOp2Blob);
  } else {
    int32_t part_id = -1;
    int32_t part_num = -1;
    std::tie(part_id, part_num) =
        GetPartIdAndPartNumFromParallelCtx(parallel_ctx);
    InitModelBlobsWithDir(ctx.device_ctx, part_id, part_num, model_load_dir,
                          BnInOp2Blob);
  }
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

void Kernel::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (kernel_conf_.need_do_col_num()) { ForwardColNum(ctx, BnInOp2Blob); }
  ForwardDataContent(ctx, BnInOp2Blob);
  if (kernel_conf_.need_do_data_id()) { ForwardDataId(ctx, BnInOp2Blob); }
}

void Kernel::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BackwardDataContent(ctx, BnInOp2Blob);
  if (HasModelBns() && Global<JobDesc>::Get()->L2() > 0.0f) {
    L2Regularization(ctx, BnInOp2Blob);
  }
  if (kernel_conf_.need_do_data_id()) { BackwardDataId(ctx, BnInOp2Blob); }
  if (kernel_conf_.need_do_col_num()) { BackwardColNum(ctx, BnInOp2Blob); }
}

bool Kernel::HasModelBns() const {
  return kernel_conf().model_bns().size() > 0;
}

template<DeviceType device_type>
void KernelIf<device_type>::ForwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyField(ctx.device_ctx, BnInOp2Blob, kernel_conf().input_bns(),
            kernel_conf().output_bns(), &Blob::CopyDataIdFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::ForwardColNum(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyField(ctx.device_ctx, BnInOp2Blob, kernel_conf().input_bns(),
            kernel_conf().output_bns(), &Blob::CopyColNumFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::BackwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<DeviceType device_type>
void KernelIf<device_type>::BackwardColNum(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyField(ctx.device_ctx, BnInOp2Blob, kernel_conf().output_diff_bns(),
            kernel_conf().input_diff_bns(), &Blob::CopyColNumFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::CopyDataId(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const Blob* from_blob, const PbRpf<std::string>& to_bns) const {
  CopyField(ctx, BnInOp2Blob, from_blob, to_bns, &Blob::CopyDataIdFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::CopyColNum(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const Blob* from_blob, const PbRpf<std::string>& to_bns) const {
  CopyField(ctx, BnInOp2Blob, from_blob, to_bns, &Blob::CopyColNumFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::CopyField(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const Blob* from_blob, const PbRpf<std::string>& to_bns,
    void (Blob::*Copy)(DeviceCtx*, const Blob*)) const {
  for (const std::string& to_bn : to_bns) {
    (BnInOp2Blob(to_bn)->*Copy)(ctx, from_blob);
  }
}

template<DeviceType device_type>
void KernelIf<device_type>::CopyField(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    const PbRpf<std::string>& from_bns, const PbRpf<std::string>& to_bns,
    void (Blob::*Copy)(DeviceCtx*, const Blob*)) const {
  if (from_bns.size() == 1) {
    const Blob* in_blob = BnInOp2Blob(from_bns[0]);
    CopyField(ctx, BnInOp2Blob, in_blob, to_bns, Copy);
  } else {
    CHECK_EQ(from_bns.size(), to_bns.size());
    FOR_RANGE(size_t, i, 0, from_bns.size()) {
      Blob* in_blob = BnInOp2Blob(from_bns[i]);
      Blob* out_blob = BnInOp2Blob(to_bns[i]);
      (out_blob->*Copy)(ctx, in_blob);
    }
  }
}

template<DeviceType device_type, typename ModelType>
void KernelIfWithModel<device_type, ModelType>::L2Regularization(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  for (const std::string& mbn : this->kernel_conf().model_bns()) {
    const Blob* model_blob = BnInOp2Blob(mbn);
    ModelType l2 = static_cast<ModelType>(
        Global<JobDesc>::Get()->L2()
        * BnInOp2Blob(this->kernel_conf().output_diff_bns()[0])->shape().At(0));
    KernelUtil<device_type, ModelType>::Axpy(
        ctx.device_ctx, static_cast<int>(model_blob->shape().elem_cnt()), l2,
        model_blob->dptr<ModelType>(), 1,
        BnInOp2Blob(GenDiffBn(mbn))->mut_dptr<ModelType>(), 1);
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
  AddKernelCreator(opcase, [creator](const KernelConf&) { return creator(); });
}

std::unique_ptr<const Kernel> ConstructKernel(
    const ParallelContext* parallel_ctx, const KernelConf& conf) {
  OperatorConf::OpTypeCase opcase = conf.op_conf().op_type_case();
  auto it = GetCreatorsMap().find(opcase);
  CHECK(it != GetCreatorsMap().end()) << opcase;
  Kernel* rptr = it->second(conf);
  rptr->Init(parallel_ctx, conf);
  return std::unique_ptr<const Kernel>(rptr);
}

#define INSTANTIATE_KERNEL_IF(device_type) template class KernelIf<device_type>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_IF, DEVICE_TYPE_SEQ);

#define INSTANTIATE_KERNEL_IF_WITH_MODEL(device_type, data_type_pair) \
  template class KernelIfWithModel<device_type,                       \
                                   OF_PP_PAIR_FIRST(data_type_pair)>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_IF_WITH_MODEL,
                                 DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
