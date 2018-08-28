#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/common/gdb.h"

namespace oneflow {

namespace {

template<typename T>
void ClearPbBlob(Blob* blob) {
  FOR_RANGE(int64_t, i, 0, blob->shape().elem_cnt()) { blob->mut_dptr<T>()[i].Clear(); }
}

void ClearPbBlob(Blob* blob) {
  switch (blob->data_type()) {
#define CLEAR_PB_BLOB_CASE(type_cpp, type_proto) \
  case type_proto: return ClearPbBlob<type_cpp>(blob);
    OF_PP_FOR_EACH_TUPLE(CLEAR_PB_BLOB_CASE, RECORD_DATA_TYPE_SEQ);
    default: CHECK_NE(blob->data_type(), DataType::kInvalidDataType);
  }
}

}  // namespace

void Kernel::Init(const ParallelContext* parallel_ctx, const KernelConf& kernel_conf,
                  DeviceCtx* device_ctx) {
  kernel_conf_ = kernel_conf;
  VirtualKernelInit(parallel_ctx, device_ctx);
}

void Kernel::InitModelAndConstBuf(const KernelCtx& ctx, const ParallelContext* parallel_ctx,
                                  const Snapshot* snapshot,
                                  std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitConstBufBlobs(ctx.device_ctx, BnInOp2Blob);
  std::string model_load_dir = "";
  if (snapshot) {
    model_load_dir = snapshot->GetDirFromOpName(op_conf().name());
  } else {
    model_load_dir = op_conf().model_load_dir();
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
    gdb::ForwardEnterBreakPoint(op_attribute(), BnInOp2Blob);
    Forward(ctx, BnInOp2Blob);
    gdb::ForwardLeaveBreakPoint(op_attribute(), BnInOp2Blob);
  } else {
    gdb::BackwardEnterBreakPoint(op_attribute(), BnInOp2Blob);
    Backward(ctx, BnInOp2Blob);
    gdb::BackwardLeaveBreakPoint(op_attribute(), BnInOp2Blob);
  }
}

const LogicalBlobId& Kernel::BnInOp2Lbi(const std::string& bn_in_op) const {
  return op_attribute().bn_in_op2lbi().at(bn_in_op);
}

void Kernel::Forward(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ClearPbBlobs(ctx, BnInOp2Blob);
  ForwardDataContent(ctx, BnInOp2Blob);
  if (GetActivationType() != ActivationType::kNone) {
    const PbRpf<std::string> obns = this->op_attribute().output_bns();
    CHECK_EQ(obns.size(), 1);

    Blob* out_blob = BnInOp2Blob(obns[0]);
    ForwardActivation(ctx, out_blob);
  }
  if (kernel_conf_.need_do_opaque_header()) {
    ForwardPackedHeader(ctx, BnInOp2Blob);
  } else {
    if (kernel_conf_.need_do_data_id()) { ForwardDataId(ctx, BnInOp2Blob); }
    if (kernel_conf_.need_do_col_num()) { ForwardColNum(ctx, BnInOp2Blob); }
  }
}

void Kernel::ClearPbBlobs(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  for (const auto& out_bn : op_attribute().pb_output_bns()) {
    Blob* out_blob = BnInOp2Blob(out_bn);
    CHECK_NOTNULL(out_blob);
    CHECK(IsRecordDataType(out_blob->data_type()));
    ClearPbBlob(out_blob);
  }
}

void Kernel::Backward(const KernelCtx& ctx,
                      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ActivationType activation = GetActivationType();
  if (activation != ActivationType::kNone) {
    const PbRpf<std::string> obns = this->op_attribute().output_bns();
    const PbRpf<std::string> odbns = this->op_attribute().output_diff_bns();
    CHECK_EQ(obns.size(), 1);
    CHECK_EQ(odbns.size(), 1);

    const Blob* out_blob = BnInOp2Blob(obns[0]);
    const Blob* out_diff_blob = BnInOp2Blob(odbns[0]);
    Blob* bw_activation_blob = BnInOp2Blob("bw_activation");
    CHECK(bw_activation_blob != nullptr);
    BackwardActivation(ctx, out_blob, out_diff_blob, bw_activation_blob);
    BackwardDataContent(ctx, [&](const std::string& bn) -> Blob* {
      if (bn == odbns[0]) {
        return bw_activation_blob;
      } else {
        return BnInOp2Blob(bn);
      }
    });
  } else {
    BackwardDataContent(ctx, BnInOp2Blob);
  }
  if (kernel_conf_.need_do_data_id()) { BackwardDataId(ctx, BnInOp2Blob); }
  if (kernel_conf_.need_do_col_num()) { BackwardColNum(ctx, BnInOp2Blob); }
}

bool Kernel::HasModelBns() const { return op_attribute().model_bns().size() > 0; }

template<DeviceType device_type>
void KernelIf<device_type>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardField(ctx.device_ctx, BnInOp2Blob, &Blob::CopyDataIdFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::ForwardColNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardField(ctx.device_ctx, BnInOp2Blob, &Blob::CopyColNumFrom);
}

template<DeviceType device_type>
void KernelIf<device_type>::ForwardField(DeviceCtx* ctx,
                                         std::function<Blob*(const std::string&)> BnInOp2Blob,
                                         void (Blob::*Copy)(DeviceCtx*, const Blob*)) const {
  const PbRpf<std::string>* input_bn = &op_attribute().input_bns();
  if (input_bn->empty()) { input_bn = &op_attribute().pb_input_bns(); }
  CopyField(ctx, BnInOp2Blob, *input_bn, op_attribute().output_bns(), Copy);
  CopyField(ctx, BnInOp2Blob, *input_bn, op_attribute().pb_output_bns(), Copy);
}

template<DeviceType device_type>
void KernelIf<device_type>::ForwardPackedHeader(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CopyField(ctx.device_ctx, BnInOp2Blob, op_attribute().input_bns(), op_attribute().output_bns(),
            &Blob::CopyHeaderFrom);
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
    CHECK(from_bns.size() == 0 || to_bns.size() == 0);
  }
}

std::unique_ptr<const Kernel> ConstructKernel(const ParallelContext* parallel_ctx,
                                              const KernelConf& conf, DeviceCtx* device_ctx) {
  Kernel* rptr = NewObj<Kernel>(conf.op_attribute().op_conf().op_type_case(), conf);
  rptr->Init(parallel_ctx, conf, device_ctx);
  return std::unique_ptr<const Kernel>(rptr);
}

template<DeviceType device_type, typename T>
ActivationType KernelIfWithActivation<device_type, T>::GetActivationType() const {
  return static_cast<ActivationType>(this->GetEnumFromCustomizedOpConf("activation"));
}

template<DeviceType device_type, typename T>
void KernelIfWithActivation<device_type, T>::ForwardActivation(const KernelCtx& ctx,
                                                               Blob* out_blob) const {
  T* out_dptr = out_blob->mut_dptr<T>();
  int64_t elem_cnt = out_blob->shape().elem_cnt();

  switch (GetActivationType()) {
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

template<DeviceType device_type, typename T>
void KernelIfWithActivation<device_type, T>::BackwardActivation(const KernelCtx& ctx,
                                                                const Blob* out_blob,
                                                                const Blob* out_diff_blob,
                                                                Blob* bw_activation_blob) const {
  int64_t elem_cnt = out_blob->shape().elem_cnt();
  switch (GetActivationType()) {
#define DEFINE_ONE_CASE(activation_type)                                    \
  case ActivationType::k##activation_type:                                  \
    KernelUtil<device_type, T>::activation_type##Backward(                  \
        ctx.device_ctx, elem_cnt, out_blob->dptr<T>(), out_blob->dptr<T>(), \
        out_diff_blob->dptr<T>(), bw_activation_blob->mut_dptr<T>());       \
    break
    DEFINE_ONE_CASE(TanH);
    DEFINE_ONE_CASE(Sigmoid);
    DEFINE_ONE_CASE(Relu);
#undef DEFINE_ONE_CASE
    default: UNIMPLEMENTED();
  }
}

#define INSTANTIATE_KERNEL_IF(device_type) template class KernelIf<device_type>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_IF, DEVICE_TYPE_SEQ);

#define INSTANTIATE_KERNEL_IF_SUBCLASS(device_type, data_type_pair)                \
  template class KernelIfWithModel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>; \
  template class KernelIfWithActivation<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_IF_SUBCLASS, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
