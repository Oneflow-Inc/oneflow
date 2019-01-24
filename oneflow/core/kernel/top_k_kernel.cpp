#include "oneflow/core/kernel/top_k_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void TopKKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* fw_buf_blob = BnInOp2Blob("fw_buf");
  Blob* out_blob = BnInOp2Blob("out");

  const int32_t instance_size = in_blob->shape().dim_vec().back();
  const int32_t instance_num = in_blob->shape().elem_cnt() / instance_size;
  const auto& conf = this->op_conf().top_k_conf();
  TopKKernelUtil<device_type, T>::Forward(in_blob->dptr<T>(), instance_num, instance_size, conf.k(),
                                          conf.sorted(), fw_buf_blob->mut_dptr<int32_t>(),
                                          out_blob->mut_dptr<int32_t>());
}

template<typename T>
struct TopKKernelUtil<DeviceType::kCPU, T> {
  static void Forward(const T* in, const int32_t instance_num, const int32_t instance_size,
                      const int32_t k, const bool sorted, int32_t* fw_buf, int32_t* out) {
    FOR_RANGE(int32_t, i, 0, instance_num) {
      std::iota(fw_buf, fw_buf + instance_size, 0);
      const int32_t offset = i * instance_size;
      auto comp = [&](const int32_t lhs, const int32_t rhs) {
        return in[offset + lhs] > in[offset + rhs];
      };
      std::nth_element(fw_buf, fw_buf + k, fw_buf + instance_size, comp);
      if (k > 1 && sorted) { std::sort(fw_buf, fw_buf + k, comp); }
      std::copy(fw_buf, fw_buf + k, out + i * k);
    }
  }
};

#define INSTANTIATE_TOP_K_KERNEL_UTIL(type_cpp, type_proto) \
  template struct TopKKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_TOP_K_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kTopKConf, TopKKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
