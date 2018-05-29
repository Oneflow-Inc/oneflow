#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

void Kernel::Init(const ParallelContext* parallel_ctx, const KernelConf& kernel_conf,
                  DeviceCtx* device_ctx) {
  kernel_conf_ = kernel_conf;
  VirtualKernelInit(parallel_ctx, device_ctx);
}

void Kernel::InitModelAndConstBuf(const KernelCtx& ctx, const ParallelContext* parallel_ctx,
                                  const Snapshot* snapshot,
                                  std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitConstBufBlobs(ctx.device_ctx, BnInOp2Blob);
  std::string model_load_dir = op_conf().model_load_dir();
  if (model_load_dir == "" && snapshot) {
    model_load_dir = snapshot->GetDirFromOpName(op_conf().name());
  }
  if (model_load_dir == "") {
    std::mt19937* random_seed_gen = static_cast<std::mt19937*>(ctx.other);
    InitModelBlobsWithRandomSeed(ctx.device_ctx, random_seed_gen, BnInOp2Blob);
  } else {
    int32_t part_id = -1;
    int32_t part_num = -1;
    std::tie(part_id, part_num) = GetPartIdAndPartNumFromParallelCtx(parallel_ctx);
    InitModelBlobsWithDir(ctx.device_ctx, part_id, part_num, model_load_dir, BnInOp2Blob);
  }
}

void Kernel::Launch(const KernelCtx& ctx,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (kernel_conf_.is_forward()) {
    Forward(ctx, BnInOp2Blob);
  } else {
    Backward(ctx, BnInOp2Blob);
  }
}

const LogicalBlobId& Kernel::BnInOp2Lbi(const std::string& bn_in_op) const {
  return op_attribute().bn_in_op2lbi().at(bn_in_op);
}

void Kernel::Forward(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (kernel_conf_.need_do_col_num()) { ForwardColNum(ctx, BnInOp2Blob); }
  ForwardDataContent(ctx, BnInOp2Blob);
  ForwardActivation(ctx, BnInOp2Blob);
  if (kernel_conf_.need_do_data_id()) { ForwardDataId(ctx, BnInOp2Blob); }
}

void Kernel::Backward(const KernelCtx& ctx,
                      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  size_t byte_size_used_by_activation = 0;
  BackwardActivation(ctx, BnInOp2Blob, &byte_size_used_by_activation);
  if (byte_size_used_by_activation > 0) {
    CHECK_NE(this->GetActivationType(), ActivationType::kNone);
    const PbRpf<std::string> odbns = this->op_attribute().output_diff_bns();
    CHECK_EQ(odbns.size(), 1);
    // generate activation_blob use buf_ptr
    Blob* out_diff_blob = BnInOp2Blob(odbns[0]);
    BlobDesc activation_blob_desc(out_diff_blob->shape(), out_diff_blob->data_type(), false, false,
                                  1);
    std::unique_ptr<Blob> activation_blob;
    GenActivationBlob(&activation_blob, ctx.device_ctx->buf_ptr(), &activation_blob_desc);
    // generate new kernel_ctx for buf_ptr and buf_size cut by activation_blob
    std::unique_ptr<DeviceCtx> device_ctx = ctx.device_ctx->CloneDeviceCtx();
    size_t aligned_offset_byte_size = RoundUp(byte_size_used_by_activation, kCudaAlignSize);
    device_ctx->set_buf_ptr(ctx.device_ctx->buf_ptr() + aligned_offset_byte_size);
    device_ctx->set_buf_size(ctx.device_ctx->buf_size() - aligned_offset_byte_size);
    CHECK_GE(device_ctx->buf_size(), 0);
    KernelCtx new_ctx;
    new_ctx.other = ctx.other;
    new_ctx.device_ctx = device_ctx.get();

    BackwardDataContent(new_ctx, [&](const std::string& bn) -> Blob* {
      if (bn == odbns[0]) { return activation_blob.get(); }
      return BnInOp2Blob(bn);
    });
  } else {
    CHECK_EQ(this->GetActivationType(), ActivationType::kNone);
    BackwardDataContent(ctx, BnInOp2Blob);
  }

  if (kernel_conf_.need_do_data_id()) { BackwardDataId(ctx, BnInOp2Blob); }
  if (kernel_conf_.need_do_col_num()) { BackwardColNum(ctx, BnInOp2Blob); }
}

bool Kernel::HasModelBns() const { return op_attribute().model_bns().size() > 0; }

template<DeviceType device_type>
void KernelIf<device_type>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyField(ctx.device_ctx, BnInOp2Blob, op_attribute().input_bns(), op_attribute().output_bns(),
            &Blob::CopyDataIdFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::ForwardColNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyField(ctx.device_ctx, BnInOp2Blob, op_attribute().input_bns(), op_attribute().output_bns(),
            &Blob::CopyColNumFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::BackwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<DeviceType device_type>
void KernelIf<device_type>::BackwardColNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyField(ctx.device_ctx, BnInOp2Blob, op_attribute().output_diff_bns(),
            op_attribute().input_diff_bns(), &Blob::CopyColNumFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::CopyDataId(DeviceCtx* ctx,
                                       std::function<Blob*(const std::string&)> BnInOp2Blob,
                                       const Blob* from_blob,
                                       const PbRpf<std::string>& to_bns) const {
  CopyField(ctx, BnInOp2Blob, from_blob, to_bns, &Blob::CopyDataIdFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::CopyColNum(DeviceCtx* ctx,
                                       std::function<Blob*(const std::string&)> BnInOp2Blob,
                                       const Blob* from_blob,
                                       const PbRpf<std::string>& to_bns) const {
  CopyField(ctx, BnInOp2Blob, from_blob, to_bns, &Blob::CopyColNumFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::CopyField(DeviceCtx* ctx,
                                      std::function<Blob*(const std::string&)> BnInOp2Blob,
                                      const Blob* from_blob, const PbRpf<std::string>& to_bns,
                                      void (Blob::*Copy)(DeviceCtx*, const Blob*)) const {
  for (const std::string& to_bn : to_bns) { (BnInOp2Blob(to_bn)->*Copy)(ctx, from_blob); }
}

template<DeviceType device_type>
void KernelIf<device_type>::CopyField(DeviceCtx* ctx,
                                      std::function<Blob*(const std::string&)> BnInOp2Blob,
                                      const PbRpf<std::string>& from_bns,
                                      const PbRpf<std::string>& to_bns,
                                      void (Blob::*Copy)(DeviceCtx*, const Blob*)) const {
  if (from_bns.size() == 1) {
    const Blob* in_blob = BnInOp2Blob(from_bns[0]);
    CopyField(ctx, BnInOp2Blob, in_blob, to_bns, Copy);
  } else if (to_bns.size() == 1) {
    Blob* in_blob = BnInOp2Blob(from_bns[0]);
    Blob* out_blob = BnInOp2Blob(to_bns[0]);
    (out_blob->*Copy)(ctx, in_blob);
  } else {
    CHECK_EQ(from_bns.size(), to_bns.size());
    FOR_RANGE(size_t, i, 0, from_bns.size()) {
      Blob* in_blob = BnInOp2Blob(from_bns[i]);
      Blob* out_blob = BnInOp2Blob(to_bns[i]);
      (out_blob->*Copy)(ctx, in_blob);
    }
  }
}

template<DeviceType device_type, typename T>
ActivationType KernelIfWithActivation<device_type, T>::GetActivationType() const {
  return static_cast<ActivationType>(this->GetEnumFromCustomizedOpConf("activation"));
}

template<DeviceType device_type, typename T>
void KernelIfWithActivation<device_type, T>::ForwardActivation(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string> obns = this->op_attribute().output_bns();
  CHECK_EQ(obns.size(), 1);
  Blob* out_blob = BnInOp2Blob(obns[0]);
  ActivationType activation = GetActivationType();
  if (activation != ActivationType::kNone) {
    T* out_dptr = out_blob->mut_dptr<T>();
    int64_t elem_cnt = out_blob->shape().elem_cnt();

    switch (activation) {
#define DEFINE_ONE_CASE(activation_type)                                                       \
  case ActivationType::k##activation_type:                                                     \
    KernelUtil<device_type, T>::activation_type(ctx.device_ctx, elem_cnt, out_dptr, out_dptr); \
    break;
      DEFINE_ONE_CASE(TanH)
      DEFINE_ONE_CASE(Sigmoid)
      DEFINE_ONE_CASE(Relu)
#undef DEFINE_ONE_CASE
      default: UNIMPLEMENTED();
    }
  }
}

template<DeviceType device_type, typename T>
void KernelIfWithActivation<device_type, T>::BackwardActivation(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
    size_t* byte_size_used_by_activation) const {
  ActivationType activation = GetActivationType();
  if (activation != ActivationType::kNone) {
    const PbRpf<std::string> obns = this->op_attribute().output_bns();
    const PbRpf<std::string> odbns = this->op_attribute().output_diff_bns();
    CHECK_EQ(obns.size(), 1);
    CHECK_EQ(odbns.size(), 1);

    const Blob* out_blob = BnInOp2Blob(obns[0]);
    const Blob* out_diff_blob = BnInOp2Blob(odbns[0]);
    *byte_size_used_by_activation = out_blob->ByteSizeOfDataContentField();
    int64_t elem_cnt = out_blob->shape().elem_cnt();

    switch (activation) {
#define DEFINE_ONE_CASE(activation_type)                                       \
  case ActivationType::k##activation_type:                                     \
    KernelUtil<device_type, T>::activation_type##Backward(                     \
        ctx.device_ctx, elem_cnt, out_blob->dptr<T>(), out_blob->dptr<T>(),    \
        out_diff_blob->dptr<T>(), static_cast<T*>(ctx.device_ctx->buf_ptr())); \
    break;
      DEFINE_ONE_CASE(TanH)
      DEFINE_ONE_CASE(Sigmoid)
      DEFINE_ONE_CASE(Relu)
#undef DEFINE_ONE_CASE
      default: UNIMPLEMENTED();
    }
  }
}

template<DeviceType device_type, typename T>
void KernelIfWithActivation<device_type, T>::GenActivationBlob(
    std::unique_ptr<Blob>* activation_blob, void* buf_ptr, BlobDesc* activation_blob_desc) const {
  activation_blob->reset(
      NewBlob(nullptr, activation_blob_desc, static_cast<char*>(buf_ptr), nullptr, device_type));
}

std::unique_ptr<const Kernel> ConstructKernel(const ParallelContext* parallel_ctx,
                                              const KernelConf& conf, DeviceCtx* device_ctx) {
  Kernel* rptr = NewObj<Kernel>(conf.op_attribute().op_conf().op_type_case(), conf);
  rptr->Init(parallel_ctx, conf, device_ctx);
  return std::unique_ptr<const Kernel>(rptr);
}

#define INSTANTIATE_KERNEL_IF(device_type) template class KernelIf<device_type>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_IF, DEVICE_TYPE_SEQ);

#define INSTANTIATE_KERNEL_IF_SUBCLASS(device_type, data_type_pair)                \
  template class KernelIfWithModel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>; \
  template class KernelIfWithActivation<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_IF_SUBCLASS, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
