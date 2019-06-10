#include "oneflow/core/kernel/where_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename CondType, typename T>
void WhereKernel<device_type, CondType, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* condition_blob = BnInOp2Blob("condition");
  const Blob* lhs_blob = BnInOp2Blob("lhs");
  const Blob* rhs_blob = BnInOp2Blob("rhs");
  Blob* out_blob = BnInOp2Blob("out");
  const Shape shape = condition_blob->shape();
  CHECK_EQ(lhs_blob->shape(), shape);
  CHECK_EQ(rhs_blob->shape(), shape);
  CHECK_EQ(out_blob->shape(), shape);
  const int64_t elem_cnt = shape.elem_cnt();

  WhereKernelUtil<device_type, CondType, T>::Forward(
      ctx.device_ctx, elem_cnt, condition_blob->dptr<CondType>(), lhs_blob->dptr<T>(),
      rhs_blob->dptr<T>(), out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename CondType, typename T>
void WhereKernel<device_type, CondType, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* condition_blob = BnInOp2Blob("condition");
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  Blob* lhs_diff_blob = BnInOp2Blob(GenDiffBn("lhs"));
  Blob* rhs_diff_blob = BnInOp2Blob(GenDiffBn("rhs"));
  const Shape shape = condition_blob->shape();
  CHECK_EQ(out_diff_blob->shape(), shape);
  CHECK_EQ(lhs_diff_blob->shape(), shape);
  CHECK_EQ(rhs_diff_blob->shape(), shape);
  const int64_t elem_cnt = shape.elem_cnt();
  WhereKernelUtil<device_type, CondType, T>::Backward(
      ctx.device_ctx, elem_cnt, condition_blob->dptr<CondType>(), out_diff_blob->dptr<T>(),
      lhs_diff_blob->mut_dptr<T>(), rhs_diff_blob->mut_dptr<T>());
}

template<typename CondType, typename T>
struct WhereKernelUtil<DeviceType::kCPU, CondType, T> {
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const CondType* condition_ptr,
                      const T* lhs_ptr, const T* rhs_ptr, T* out_ptr) {
    UNIMPLEMENTED();
  }

  static void Backward(DeviceCtx* ctx, const int64_t elem_cnt, const CondType* condition_ptr,
                       const T* out_diff_ptr, T* lhs_diff_ptr, T* rhs_diff_ptr) {
    UNIMPLEMENTED();
  }
};

namespace {

Kernel* CreateWhereKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define WHERE_KERNEL_ENTRY(device_type, cond_type_pair, value_type_pair)                           \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(cond_type_pair), OF_PP_PAIR_SECOND(value_type_pair)), \
   []() {                                                                                          \
     return new WhereKernel<device_type, OF_PP_PAIR_FIRST(cond_type_pair),                         \
                            OF_PP_PAIR_FIRST(value_type_pair)>();                                  \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(WHERE_KERNEL_ENTRY, DEVICE_TYPE_SEQ, INT_DATA_TYPE_SEQ,
                                       ARITHMETIC_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                                kernel_conf.where_conf().cond_type(),
                                kernel_conf.where_conf().value_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kWhereConf, CreateWhereKernel);

#define MAKE_ENTRY(cond_type_pair, value_type_pair)                                   \
  template struct WhereKernelUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(cond_type_pair), \
                                  OF_PP_PAIR_FIRST(value_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, INT_DATA_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
