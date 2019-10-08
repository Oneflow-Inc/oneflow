#include "oneflow/core/kernel/cast_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename U>
void CopyBlob(DeviceCtx* ctx, const Blob* src, Blob* dst) {
  CHECK_EQ(src->shape(), dst->shape());
  if (device_type == DeviceType::kCPU) {
    CopyElem(src->dptr<T>(), dst->mut_dptr<U>(), src->shape().elem_cnt());
  } else if (device_type == DeviceType::kGPU) {
    CopyElemOnGpu(ctx, src->dptr<T>(), dst->mut_dptr<U>(), src->shape().elem_cnt());
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

#define MAKE_CASE_HANDLER_ENTRY(in_type_pair, out_type_pair)                          \
  {std::make_pair(OF_PP_PAIR_SECOND(in_type_pair), OF_PP_PAIR_SECOND(out_type_pair)), \
   CopyBlob<device_type, OF_PP_PAIR_FIRST(in_type_pair), OF_PP_PAIR_FIRST(out_type_pair)>},

template<DeviceType device_type>
struct CastUtil final {
  static void SwitchCopyBlob(const std::pair<DataType, DataType>& key, DeviceCtx* ctx,
                             const Blob* src, Blob* dst) {
    static const std::map<std::pair<DataType, DataType>,
                          std::function<void(DeviceCtx*, const Blob*, Blob*)>>
        case_handler{
            // clang-format off
          OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_CASE_HANDLER_ENTRY, POD_DATA_TYPE_SEQ, POD_DATA_TYPE_SEQ)
          MAKE_CASE_HANDLER_ENTRY((float, DataType::kFloat), (float16, DataType::kFloat16))
          MAKE_CASE_HANDLER_ENTRY((float16, DataType::kFloat16), (float, DataType::kFloat))
            // clang-format on
        };
    case_handler.at(key)(ctx, src, dst);
  }
};

template<DeviceType device_type>
void CastKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CastUtil<device_type>::SwitchCopyBlob(std::make_pair(in_blob->data_type(), out_blob->data_type()),
                                        ctx.device_ctx, in_blob, out_blob);
}

template<DeviceType device_type>
void CastKernel<device_type>::ForwardLoD(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->tree_lod_mut_view().UpdateLoD(BnInOp2Blob("in")->tree_lod_view().lod_tree());
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kCastConf, DeviceType::kCPU,
                            CastKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kCastConf, DeviceType::kGPU,
                            CastKernel<DeviceType::kGPU>);

}  // namespace oneflow
