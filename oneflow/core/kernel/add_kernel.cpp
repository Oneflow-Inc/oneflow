#include "oneflow/core/kernel/add_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
namespace oneflow {

template<DeviceType device_type, typename T>
struct AddKernelUtil {
  // out += in
  static void AdditionAssign(DeviceCtx* device_ctx, const int64_t elem_cnt, Blob* out,
                             const Blob* in);
};

template<DeviceType device_type, typename T>
void AddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& ibns = this->op_attribute().input_bns();
  size_t in_num = ibns.size();
  if (in_num == 0) return;
  Blob* out_blob = BnInOp2Blob(this->op_attribute().output_bns(0));
  Memset<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(), 0,
                      out_blob->ByteSizeOfDataContentField());
  auto in_blob = [&](int32_t idx) { return BnInOp2Blob(this->op_attribute().input_bns(idx)); };
  int32_t offset = 0;
  while (in_num - offset >= 10) {
    AddKernelUtil<device_type, T>::AdditionAssign(
        ctx.device_ctx, out_blob, in_blob(offset), in_blob(offset + 1), in_blob(offset + 2),
        in_blob(offset + 3), in_blob(offset + 4), in_blob(offset + 5), in_blob(offset + 6),
        in_blob(offset + 7), in_blob(offset + 8), in_blob(offset + 9));
    offset += 10;
  }

  if (in_num - offset > 0) {
    switch (in_num - offset) {
      case 1:
        AddKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, out_blob, in_blob(offset));
        break;
      case 2:
        AddKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, out_blob, in_blob(offset),
                                                      in_blob(offset + 1));
        break;
      case 3:
        AddKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, out_blob, in_blob(offset),
                                                      in_blob(offset + 1), in_blob(offset + 2));
        break;
      case 4:
        AddKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, out_blob, in_blob(offset),
                                                      in_blob(offset + 1), in_blob(offset + 2),
                                                      in_blob(offset + 3));
        break;
      case 5:
        AddKernelUtil<device_type, T>::AdditionAssign(ctx.device_ctx, out_blob, in_blob(offset),
                                                      in_blob(offset + 1), in_blob(offset + 2),
                                                      in_blob(offset + 3), in_blob(offset + 4));
        break;
      case 6:
        AddKernelUtil<device_type, T>::AdditionAssign(
            ctx.device_ctx, out_blob, in_blob(offset), in_blob(offset + 1), in_blob(offset + 2),
            in_blob(offset + 3), in_blob(offset + 4), in_blob(offset + 5));
        break;
      case 7:
        AddKernelUtil<device_type, T>::AdditionAssign(
            ctx.device_ctx, out_blob, in_blob(offset), in_blob(offset + 1), in_blob(offset + 2),
            in_blob(offset + 3), in_blob(offset + 4), in_blob(offset + 5), in_blob(offset + 6));
        break;
      case 8:
        AddKernelUtil<device_type, T>::AdditionAssign(
            ctx.device_ctx, out_blob, in_blob(offset), in_blob(offset + 1), in_blob(offset + 2),
            in_blob(offset + 3), in_blob(offset + 4), in_blob(offset + 5), in_blob(offset + 6),
            in_blob(offset + 7));
        break;
      case 9:
        AddKernelUtil<device_type, T>::AdditionAssign(
            ctx.device_ctx, out_blob, in_blob(offset), in_blob(offset + 1), in_blob(offset + 2),
            in_blob(offset + 3), in_blob(offset + 4), in_blob(offset + 5), in_blob(offset + 6),
            in_blob(offset + 7), in_blob(offset + 8));
        break;
    }
  }
}

template<DeviceType device_type, typename T>
void AddKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  FOR_RANGE(size_t, i, 0, this->op_attribute().input_diff_bns().size()) {
    Blob* in_diff_blob = BnInOp2Blob(this->op_attribute().input_diff_bns(i));
    in_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
  }
}

template<DeviceType device_type, typename T>
const PbMessage& AddKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().add_conf();
}

#define DEFINE_FLOATING_ADD_KERNEL_UTIL(type_cpp, type_proto)                                      \
  template<DeviceType device_type>                                                                 \
  struct AddKernelUtil<device_type, type_cpp> {                                                    \
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

OF_PP_FOR_EACH_TUPLE(DEFINE_FLOATING_ADD_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

#define DEFINE_NONFLOAT_ADD_KERNEL_UTIL(type_cpp, type_proto)                          \
  template<DeviceType device_type>                                                     \
  struct AddKernelUtil<device_type, type_cpp> {                                        \
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

OF_PP_FOR_EACH_TUPLE(DEFINE_NONFLOAT_ADD_KERNEL_UTIL, INT_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAddConf, AddKernel, ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
