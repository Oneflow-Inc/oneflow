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
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  if (out_num > 10) {
    for (size_t i = 0; i != odbns.size(); ++i) {
      const Blob* out_diff_blob = BnInOp2Blob(odbns[i]);
      CloneKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, in_diff_blob, out_diff_blob);
    }
  } else {
    switch (out_num) {
      case 1: {
        Blob* out_diff_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        CloneKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, in_diff_blob, out_diff_0);
      } break;
      case 2: {
        Blob* out_diff_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_diff_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        CloneKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, in_diff_blob, out_diff_0,
                                                        out_diff_1);
      } break;
      case 3: {
        Blob* out_diff_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_diff_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_diff_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        CloneKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, in_diff_blob, out_diff_0,
                                                        out_diff_1, out_diff_2);
      } break;
      case 4: {
        Blob* out_diff_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_diff_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_diff_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_diff_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        CloneKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, in_diff_blob, out_diff_0,
                                                        out_diff_1, out_diff_2, out_diff_3);
      } break;
      case 5: {
        Blob* out_diff_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_diff_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_diff_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_diff_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        Blob* out_diff_4 = BnInOp2Blob(this->op_attribute().output_diff_bns(4));
        CloneKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, in_diff_blob, out_diff_0,
                                                        out_diff_1, out_diff_2, out_diff_3,
                                                        out_diff_4);
      } break;
      case 6: {
        Blob* out_diff_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_diff_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_diff_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_diff_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        Blob* out_diff_4 = BnInOp2Blob(this->op_attribute().output_diff_bns(4));
        Blob* out_diff_5 = BnInOp2Blob(this->op_attribute().output_diff_bns(5));
        CloneKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, in_diff_blob, out_diff_0,
                                                        out_diff_1, out_diff_2, out_diff_3,
                                                        out_diff_4, out_diff_5);
      } break;
      case 7: {
        Blob* out_diff_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_diff_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_diff_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_diff_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        Blob* out_diff_4 = BnInOp2Blob(this->op_attribute().output_diff_bns(4));
        Blob* out_diff_5 = BnInOp2Blob(this->op_attribute().output_diff_bns(5));
        Blob* out_diff_6 = BnInOp2Blob(this->op_attribute().output_diff_bns(6));
        CloneKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, in_diff_blob, out_diff_0,
                                                        out_diff_1, out_diff_2, out_diff_3,
                                                        out_diff_4, out_diff_5, out_diff_6);
      } break;
      case 8: {
        Blob* out_diff_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_diff_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_diff_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_diff_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        Blob* out_diff_4 = BnInOp2Blob(this->op_attribute().output_diff_bns(4));
        Blob* out_diff_5 = BnInOp2Blob(this->op_attribute().output_diff_bns(5));
        Blob* out_diff_6 = BnInOp2Blob(this->op_attribute().output_diff_bns(6));
        Blob* out_diff_7 = BnInOp2Blob(this->op_attribute().output_diff_bns(7));
        CloneKernelUtil<device_type, T>::AdditionAssign(
            ctx.device_ctx, in_diff_blob, out_diff_0, out_diff_1, out_diff_2, out_diff_3,
            out_diff_4, out_diff_5, out_diff_6, out_diff_7);
      } break;
      case 9: {
        Blob* out_diff_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_diff_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_diff_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_diff_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        Blob* out_diff_4 = BnInOp2Blob(this->op_attribute().output_diff_bns(4));
        Blob* out_diff_5 = BnInOp2Blob(this->op_attribute().output_diff_bns(5));
        Blob* out_diff_6 = BnInOp2Blob(this->op_attribute().output_diff_bns(6));
        Blob* out_diff_7 = BnInOp2Blob(this->op_attribute().output_diff_bns(7));
        Blob* out_diff_8 = BnInOp2Blob(this->op_attribute().output_diff_bns(8));
        CloneKernelUtil<device_type, T>::AdditionAssign(
            ctx.device_ctx, in_diff_blob, out_diff_0, out_diff_1, out_diff_2, out_diff_3,
            out_diff_4, out_diff_5, out_diff_6, out_diff_7, out_diff_8);
      } break;
      case 10: {
        Blob* out_diff_0 = BnInOp2Blob(this->op_attribute().output_diff_bns(0));
        Blob* out_diff_1 = BnInOp2Blob(this->op_attribute().output_diff_bns(1));
        Blob* out_diff_2 = BnInOp2Blob(this->op_attribute().output_diff_bns(2));
        Blob* out_diff_3 = BnInOp2Blob(this->op_attribute().output_diff_bns(3));
        Blob* out_diff_4 = BnInOp2Blob(this->op_attribute().output_diff_bns(4));
        Blob* out_diff_5 = BnInOp2Blob(this->op_attribute().output_diff_bns(5));
        Blob* out_diff_6 = BnInOp2Blob(this->op_attribute().output_diff_bns(6));
        Blob* out_diff_7 = BnInOp2Blob(this->op_attribute().output_diff_bns(7));
        Blob* out_diff_8 = BnInOp2Blob(this->op_attribute().output_diff_bns(8));
        Blob* out_diff_9 = BnInOp2Blob(this->op_attribute().output_diff_bns(9));
        CloneKernelUtil<device_type, T>::AdditionAssign(
            ctx.device_ctx, in_diff_blob, out_diff_0, out_diff_1, out_diff_2, out_diff_3,
            out_diff_4, out_diff_5, out_diff_6, out_diff_7, out_diff_8, out_diff_9);
      } break;
    }
  }
}

#define DEFINE_FLOATING_CLONE_KERNEL_UTIL(type_cpp, type_proto)                                    \
  template<DeviceType device_type>                                                                 \
  struct CloneKernelUtil<device_type, type_cpp> {                                                  \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0) {               \
      KernelUtil<device_type, type_cpp>::AdditionAssign(                                           \
          device_ctx, out->shape().elem_cnt(), out->mut_dptr<type_cpp>(), in_0->dptr<type_cpp>()); \
    }                                                                                              \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,                 \
                               const Blob* in_1) {                                                 \
      KernelUtil<device_type, type_cpp>::AdditionAssign(                                           \
          device_ctx, out->shape().elem_cnt(), out->mut_dptr<type_cpp>(), in_0->dptr<type_cpp>(),  \
          in_1->dptr<type_cpp>());                                                                 \
    }                                                                                              \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,                 \
                               const Blob* in_1, const Blob* in_2) {                               \
      KernelUtil<device_type, type_cpp>::AdditionAssign(                                           \
          device_ctx, out->shape().elem_cnt(), out->mut_dptr<type_cpp>(), in_0->dptr<type_cpp>(),  \
          in_1->dptr<type_cpp>(), in_2->dptr<type_cpp>());                                         \
    }                                                                                              \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,                 \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3) {             \
      KernelUtil<device_type, type_cpp>::AdditionAssign(                                           \
          device_ctx, out->shape().elem_cnt(), out->mut_dptr<type_cpp>(), in_0->dptr<type_cpp>(),  \
          in_1->dptr<type_cpp>(), in_2->dptr<type_cpp>(), in_3->dptr<type_cpp>());                 \
    }                                                                                              \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,                 \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3,               \
                               const Blob* in_4) {                                                 \
      KernelUtil<device_type, type_cpp>::AdditionAssign(                                           \
          device_ctx, out->shape().elem_cnt(), out->mut_dptr<type_cpp>(), in_0->dptr<type_cpp>(),  \
          in_1->dptr<type_cpp>(), in_2->dptr<type_cpp>(), in_3->dptr<type_cpp>(),                  \
          in_4->dptr<type_cpp>());                                                                 \
    }                                                                                              \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,                 \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3,               \
                               const Blob* in_4, const Blob* in_5) {                               \
      KernelUtil<device_type, type_cpp>::AdditionAssign(                                           \
          device_ctx, out->shape().elem_cnt(), out->mut_dptr<type_cpp>(), in_0->dptr<type_cpp>(),  \
          in_1->dptr<type_cpp>(), in_2->dptr<type_cpp>(), in_3->dptr<type_cpp>(),                  \
          in_4->dptr<type_cpp>(), in_5->dptr<type_cpp>());                                         \
    }                                                                                              \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,                 \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3,               \
                               const Blob* in_4, const Blob* in_5, const Blob* in_6) {             \
      KernelUtil<device_type, type_cpp>::AdditionAssign(                                           \
          device_ctx, out->shape().elem_cnt(), out->mut_dptr<type_cpp>(), in_0->dptr<type_cpp>(),  \
          in_1->dptr<type_cpp>(), in_2->dptr<type_cpp>(), in_3->dptr<type_cpp>(),                  \
          in_4->dptr<type_cpp>(), in_5->dptr<type_cpp>(), in_6->dptr<type_cpp>());                 \
    }                                                                                              \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,                 \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3,               \
                               const Blob* in_4, const Blob* in_5, const Blob* in_6,               \
                               const Blob* in_7) {                                                 \
      KernelUtil<device_type, type_cpp>::AdditionAssign(                                           \
          device_ctx, out->shape().elem_cnt(), out->mut_dptr<type_cpp>(), in_0->dptr<type_cpp>(),  \
          in_1->dptr<type_cpp>(), in_2->dptr<type_cpp>(), in_3->dptr<type_cpp>(),                  \
          in_4->dptr<type_cpp>(), in_5->dptr<type_cpp>(), in_6->dptr<type_cpp>(),                  \
          in_7->dptr<type_cpp>());                                                                 \
    }                                                                                              \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,                 \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3,               \
                               const Blob* in_4, const Blob* in_5, const Blob* in_6,               \
                               const Blob* in_7, const Blob* in_8) {                               \
      KernelUtil<device_type, type_cpp>::AdditionAssign(                                           \
          device_ctx, out->shape().elem_cnt(), out->mut_dptr<type_cpp>(), in_0->dptr<type_cpp>(),  \
          in_1->dptr<type_cpp>(), in_2->dptr<type_cpp>(), in_3->dptr<type_cpp>(),                  \
          in_4->dptr<type_cpp>(), in_5->dptr<type_cpp>(), in_6->dptr<type_cpp>(),                  \
          in_7->dptr<type_cpp>(), in_8->dptr<type_cpp>());                                         \
    }                                                                                              \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,                 \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3,               \
                               const Blob* in_4, const Blob* in_5, const Blob* in_6,               \
                               const Blob* in_7, const Blob* in_8, const Blob* in_9) {             \
      KernelUtil<device_type, type_cpp>::AdditionAssign(                                           \
          device_ctx, out->shape().elem_cnt(), out->mut_dptr<type_cpp>(), in_0->dptr<type_cpp>(),  \
          in_1->dptr<type_cpp>(), in_2->dptr<type_cpp>(), in_3->dptr<type_cpp>(),                  \
          in_4->dptr<type_cpp>(), in_5->dptr<type_cpp>(), in_6->dptr<type_cpp>(),                  \
          in_7->dptr<type_cpp>(), in_8->dptr<type_cpp>(), in_9->dptr<type_cpp>());                 \
    }                                                                                              \
  };

OF_PP_FOR_EACH_TUPLE(DEFINE_FLOATING_CLONE_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

#define DEFINE_NONFLOAT_CLONE_KERNEL_UTIL(type_cpp, type_proto)                        \
  template<DeviceType device_type>                                                     \
  struct CloneKernelUtil<device_type, type_cpp> {                                      \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0) {   \
      UNIMPLEMENTED();                                                                 \
    }                                                                                  \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,     \
                               const Blob* in_1) {                                     \
      UNIMPLEMENTED();                                                                 \
    }                                                                                  \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,     \
                               const Blob* in_1, const Blob* in_2) {                   \
      UNIMPLEMENTED();                                                                 \
    }                                                                                  \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,     \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3) { \
      UNIMPLEMENTED();                                                                 \
    }                                                                                  \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,     \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3,   \
                               const Blob* in_4) {                                     \
      UNIMPLEMENTED();                                                                 \
    }                                                                                  \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,     \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3,   \
                               const Blob* in_4, const Blob* in_5) {                   \
      UNIMPLEMENTED();                                                                 \
    }                                                                                  \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,     \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3,   \
                               const Blob* in_4, const Blob* in_5, const Blob* in_6) { \
      UNIMPLEMENTED();                                                                 \
    }                                                                                  \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,     \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3,   \
                               const Blob* in_4, const Blob* in_5, const Blob* in_6,   \
                               const Blob* in_7) {                                     \
      UNIMPLEMENTED();                                                                 \
    }                                                                                  \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,     \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3,   \
                               const Blob* in_4, const Blob* in_5, const Blob* in_6,   \
                               const Blob* in_7, const Blob* in_8) {                   \
      UNIMPLEMENTED();                                                                 \
    }                                                                                  \
    static void AdditionAssign(DeviceCtx* device_ctx, Blob* out, const Blob* in_0,     \
                               const Blob* in_1, const Blob* in_2, const Blob* in_3,   \
                               const Blob* in_4, const Blob* in_5, const Blob* in_6,   \
                               const Blob* in_7, const Blob* in_8, const Blob* in_9) { \
      UNIMPLEMENTED();                                                                 \
    }                                                                                  \
  };

OF_PP_FOR_EACH_TUPLE(DEFINE_NONFLOAT_CLONE_KERNEL_UTIL, INT_DATA_TYPE_SEQ CHAR_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kCloneConf, CloneKernel, POD_DATA_TYPE_SEQ);

}  // namespace oneflow
