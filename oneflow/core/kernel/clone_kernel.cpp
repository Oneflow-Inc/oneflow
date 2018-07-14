#include "oneflow/core/kernel/clone_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void CloneKernel<device_type, T>::Forward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns(0));
  for (const std::string& obn : this->op_attribute().output_bns()) {
    Blob* out_blob = BnInOp2Blob(obn);
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_memory_ptr(), in_blob->memory_ptr(),
                        in_blob->TotalByteSize());
  }
}

template<DeviceType device_type, typename T>
struct CloneKernelUtil {
  // b += a
  static void AdditionAssign(DeviceCtx* device_ctx, const Blob* a, Blob* b);
};

template<DeviceType device_type, typename T>
void CloneKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& odbns = this->op_attribute().output_diff_bns();
  size_t out_num = odbns.size();
  if (out_num == 0) return;
  Blob* in_diff_blob = BnInOp2Blob(this->op_attribute().input_diff_bns(0));
  if (out_num > 10) {
    const Blob* out_diff_blob_0 = BnInOp2Blob(odbns[0]);
    Memcpy<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr(), out_diff_blob_0->dptr(),
                        out_diff_blob_0->ByteSizeOfDataContentField());
    for (size_t i = 1; i != odbns.size(); ++i) {
      const Blob* out_diff_blob = BnInOp2Blob(odbns[i]);
      CloneKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, out_diff_blob, in_diff_blob);
    }
  } else {
    const int64_t elem_cnt = in_diff_blob->shape().elem_cnt();
    switch (out_num) {
      case 1: {
        Blob* out_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        KernelUtil<device_type, T>::Add1(ctx.device_ctx, elem_cnt, in_diff_blob->mut_dptr<T>(),
                                         out_0->dptr<T>());
      } break;
      case 2: {
        Blob* out_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        KernelUtil<device_type, T>::Add2(ctx.device_ctx, elem_cnt, in_diff_blob->mut_dptr<T>(),
                                         out_0->dptr<T>(), out_1->dptr<T>());
      } break;
      case 3: {
        Blob* out_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        KernelUtil<device_type, T>::Add3(ctx.device_ctx, elem_cnt, in_diff_blob->mut_dptr<T>(),
                                         out_0->dptr<T>(), out_1->dptr<T>(), out_2->dptr<T>());
      } break;
      case 4: {
        Blob* out_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        KernelUtil<device_type, T>::Add4(ctx.device_ctx, elem_cnt, in_diff_blob->mut_dptr<T>(),
                                         out_0->dptr<T>(), out_1->dptr<T>(), out_2->dptr<T>(),
                                         out_3->dptr<T>());
      } break;
      case 5: {
        Blob* out_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        Blob* out_4 = BnInOp2Blob(this->op_attribute().output_diff_bns(4));
        KernelUtil<device_type, T>::Add5(ctx.device_ctx, elem_cnt, in_diff_blob->mut_dptr<T>(),
                                         out_0->dptr<T>(), out_1->dptr<T>(), out_2->dptr<T>(),
                                         out_3->dptr<T>(), out_4->dptr<T>());
      } break;
      case 6: {
        Blob* out_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        Blob* out_4 = BnInOp2Blob(this->op_attribute().output_diff_bns(4));
        Blob* out_5 = BnInOp2Blob(this->op_attribute().output_diff_bns(5));
        KernelUtil<device_type, T>::Add6(ctx.device_ctx, elem_cnt, in_diff_blob->mut_dptr<T>(),
                                         out_0->dptr<T>(), out_1->dptr<T>(), out_2->dptr<T>(),
                                         out_3->dptr<T>(), out_4->dptr<T>(), out_5->dptr<T>());
      } break;
      case 7: {
        Blob* out_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        Blob* out_4 = BnInOp2Blob(this->op_attribute().output_diff_bns(4));
        Blob* out_5 = BnInOp2Blob(this->op_attribute().output_diff_bns(5));
        Blob* out_6 = BnInOp2Blob(this->op_attribute().output_diff_bns(6));
        KernelUtil<device_type, T>::Add7(ctx.device_ctx, elem_cnt, in_diff_blob->mut_dptr<T>(),
                                         out_0->dptr<T>(), out_1->dptr<T>(), out_2->dptr<T>(),
                                         out_3->dptr<T>(), out_4->dptr<T>(), out_5->dptr<T>(),
                                         out_6->dptr<T>());
      } break;
      case 8: {
        Blob* out_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        Blob* out_4 = BnInOp2Blob(this->op_attribute().output_diff_bns(4));
        Blob* out_5 = BnInOp2Blob(this->op_attribute().output_diff_bns(5));
        Blob* out_6 = BnInOp2Blob(this->op_attribute().output_diff_bns(6));
        Blob* out_7 = BnInOp2Blob(this->op_attribute().output_diff_bns(7));
        KernelUtil<device_type, T>::Add8(ctx.device_ctx, elem_cnt, in_diff_blob->mut_dptr<T>(),
                                         out_0->dptr<T>(), out_1->dptr<T>(), out_2->dptr<T>(),
                                         out_3->dptr<T>(), out_4->dptr<T>(), out_5->dptr<T>(),
                                         out_6->dptr<T>(), out_7->dptr<T>());
      } break;
      case 9: {
        Blob* out_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        Blob* out_4 = BnInOp2Blob(this->op_attribute().output_diff_bns(4));
        Blob* out_5 = BnInOp2Blob(this->op_attribute().output_diff_bns(5));
        Blob* out_6 = BnInOp2Blob(this->op_attribute().output_diff_bns(6));
        Blob* out_7 = BnInOp2Blob(this->op_attribute().output_diff_bns(7));
        Blob* out_8 = BnInOp2Blob(this->op_attribute().output_diff_bns(8));
        KernelUtil<device_type, T>::Add9(ctx.device_ctx, elem_cnt, in_diff_blob->mut_dptr<T>(),
                                         out_0->dptr<T>(), out_1->dptr<T>(), out_2->dptr<T>(),
                                         out_3->dptr<T>(), out_4->dptr<T>(), out_5->dptr<T>(),
                                         out_6->dptr<T>(), out_7->dptr<T>(), out_8->dptr<T>());
      } break;
      case 10: {
        Blob* out_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        Blob* out_4 = BnInOp2Blob(this->op_attribute().output_diff_bns(4));
        Blob* out_5 = BnInOp2Blob(this->op_attribute().output_diff_bns(5));
        Blob* out_6 = BnInOp2Blob(this->op_attribute().output_diff_bns(6));
        Blob* out_7 = BnInOp2Blob(this->op_attribute().output_diff_bns(7));
        Blob* out_8 = BnInOp2Blob(this->op_attribute().output_diff_bns(8));
        Blob* out_9 = BnInOp2Blob(this->op_attribute().output_diff_bns(9));
        KernelUtil<device_type, T>::Add10(ctx.device_ctx, elem_cnt, in_diff_blob->mut_dptr<T>(),
                                          out_0->dptr<T>(), out_1->dptr<T>(), out_2->dptr<T>(),
                                          out_3->dptr<T>(), out_4->dptr<T>(), out_5->dptr<T>(),
                                          out_6->dptr<T>(), out_7->dptr<T>(), out_8->dptr<T>(),
                                          out_9->dptr<T>());
      } break;
    }
  }
}

#define DEFINE_FLOATING_CLONE_KERNEL_UTIL(type_cpp, type_proto)                                    \
  template<DeviceType device_type>                                                                 \
  struct CloneKernelUtil<device_type, type_cpp> {                                                  \
    static void AdditionAssign(DeviceCtx* device_ctx, const Blob* a, Blob* b) {                    \
      KernelUtil<device_type, type_cpp>::Axpy(device_ctx, a->shape().elem_cnt(), 1.0,              \
                                              a->dptr<type_cpp>(), 1, b->mut_dptr<type_cpp>(), 1); \
    }                                                                                              \
  };

OF_PP_FOR_EACH_TUPLE(DEFINE_FLOATING_CLONE_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

#define DEFINE_NONFLOAT_CLONE_KERNEL_UTIL(type_cpp, type_proto)                                    \
  template<DeviceType device_type>                                                                 \
  struct CloneKernelUtil<device_type, type_cpp> {                                                  \
    static void AdditionAssign(DeviceCtx* device_ctx, const Blob* a, Blob* b) { UNIMPLEMENTED(); } \
  };

OF_PP_FOR_EACH_TUPLE(DEFINE_NONFLOAT_CLONE_KERNEL_UTIL, INT_DATA_TYPE_SEQ CHAR_DATA_TYPE_SEQ)

// ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kCloneConf, CloneKernel, POD_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kCloneConf, CloneKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
