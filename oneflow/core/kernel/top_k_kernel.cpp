#include "oneflow/core/kernel/top_k_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void TopKKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* fw_buf_blob = BnInOp2Blob("fw_buf");
  Blob* out_blob = BnInOp2Blob("out");

  const int32_t instance_dim = in_blob->shape().dim_vec().back();
  const int32_t instance_num = in_blob->shape().elem_cnt() / instance_dim;
  const auto& conf = this->op_conf().top_k_conf();
  TopKKernelUtil<device_type, T>::Forward(in_blob->dptr<T>(), conf.sorted(), instance_num,
                                          instance_dim, conf.k(), fw_buf_blob->mut_dptr<int32_t>(),
                                          out_blob->mut_dptr<int32_t>());
}

template<typename T>
struct TopKKernelUtil<DeviceType::kCPU, T> {
  static void Forward(const T* in_ptr, const bool sorted, const int32_t instance_num,
                      const int32_t instance_dim, const int32_t k, int32_t* fw_buf,
                      int32_t* out_ptr) {
    FOR_RANGE(int32_t, i, 0, instance_num) {
      std::iota(fw_buf, fw_buf + instance_dim, 0);
      std::nth_element(
          fw_buf, fw_buf + k - 1, fw_buf + instance_dim,
          [&](const int32_t lhs, const int32_t rhs) { return in_ptr[lhs] > in_ptr[rhs]; });
      if (sorted) {
        std::sort(fw_buf, fw_buf + k - 1,
                  [&](const int32_t lhs, const int32_t rhs) { return in_ptr[lhs] > in_ptr[rhs]; });
      }
      FOR_RANGE(int32_t, j, 0, k) { out_ptr[i * k + j] = fw_buf[j]; }
    }
  }
};

#define INSTANTIATE_TOP_K_KERNEL_UTIL(type_cpp, type_proto) \
  template struct TopKKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_TOP_K_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

namespace {

Kernel* CreateTopKKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define TOPK_KERNEL_ENTRY(device_type, data_type_pair)         \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair)), \
   []() { return new TopKKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>(); }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(TOPK_KERNEL_ENTRY, DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                                kernel_conf.top_k_conf().data_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kTopKConf, CreateTopKKernel);

}  // namespace oneflow
