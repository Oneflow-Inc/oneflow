#include "oneflow/core/kernel/add_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

template<DeviceType device_type, typename T>
void AddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  AddUtil<device_type, T>::Add(ctx, this, BnInOp2Blob);
}

template<DeviceType device_type, typename T>
const PbMessage& AddKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().add_conf();
}

template<DeviceType device_type, typename T>
struct AddUtil {
  static void Add(const KernelCtx& ctx, const AddKernel<device_type, T>* add_kernel,
                  std::function<Blob*(const std::string&)> BnInOp2Blob) {
    const PbRpf<std::string>& ibns = add_kernel->op_attribute().input_bns();
    size_t in_num = ibns.size();
    if (in_num == 0) return;
    Blob* out_blob = BnInOp2Blob(add_kernel->op_attribute().output_bns(0));
    auto in_blob = [=](int32_t idx) {
      return BnInOp2Blob(add_kernel->op_attribute().input_bns(idx));
    };
    static const int kWidth = 8;
    int r = in_num % kWidth;
    if (r) {
      tuple_switch(r, add_kernel->tp_,
                   AdditionFunction<device_type, T, decltype(add_kernel)>{
                       out_blob, BnInOp2Blob, ctx.device_ctx, 0, add_kernel});
    }
    for (; r < in_num; r += kWidth) {
      if (r == 0) {
        Addition<device_type, T>(ctx.device_ctx, out_blob, in_blob(r), in_blob(r + 1),
                                 in_blob(r + 2), in_blob(r + 3), in_blob(r + 4), in_blob(r + 5),
                                 in_blob(r + 6), in_blob(r + 7));
      } else {
        Addition<device_type, T>(ctx.device_ctx, out_blob, out_blob, in_blob(r), in_blob(r + 1),
                                 in_blob(r + 2), in_blob(r + 3), in_blob(r + 4), in_blob(r + 5),
                                 in_blob(r + 6), in_blob(r + 7));
      }
    }
  }
};

template<>
struct AddUtil<DeviceType::kGPU, float16> {
  static void Add(const KernelCtx& ctx, const AddKernel<DeviceType::kGPU, float16>* add_kernel,
                  std::function<Blob*(const std::string&)> BnInOp2Blob) {
    const PbRpf<std::string>& ibns = add_kernel->op_attribute().input_bns();
    CHECK_GE(ibns.size(), 2);
    CHECK_LE(ibns.size(), 4);

    Blob* out = BnInOp2Blob(add_kernel->op_attribute().output_bns(0));
    float16* out_dptr = out->mut_dptr<float16>();
    std::vector<const float16*> in_dptrs;
    for (const std::string& ibn : ibns) { in_dptrs.push_back(BnInOp2Blob(ibn)->dptr<float16>()); }
    HalfGpuAdd(ctx.device_ctx, out->shape().elem_cnt(), out_dptr, in_dptrs);
  }
};

ADD_DEFAULT_KERNEL_CREATOR_WITH_GPU_HALF(OperatorConf::kAddConf, AddKernel,
                                         ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
