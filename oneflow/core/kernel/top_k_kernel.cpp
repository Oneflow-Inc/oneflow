#include "oneflow/core/kernel/top_k_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void TopKKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* fw_buf_blob = BnInOp2Blob("fw_buf");
  Blob* indices_blob = BnInOp2Blob("indices");
  Blob* values_blob = BnInOp2Blob("values");
  const auto& conf = this->op_conf().top_k_conf();

  const int32_t elem_cnt_per_instance = in_blob->shape().dim_vec().back();
  const int32_t instance_num = in_blob->shape().elem_cnt() / elem_cnt_per_instance;

  int32_t* indices_ptr = indices_blob ? indices_blob->mut_dptr<int32_t>() : nullptr;
  T* values_ptr = values_blob ? values_blob->mut_dptr<T>() : nullptr;
  TopKKernelUtil<device_type, T>::Forward(
      in_blob->dptr<T>(), conf.sorted(), instance_num, elem_cnt_per_instance, conf.k(),
      fw_buf_blob->mut_dptr<int32_t>(), indices_ptr, values_ptr);
}

template<typename T>
struct TopKKernelUtil<DeviceType::kCPU, T> {
  static void Forward(const T* in_ptr, const bool sorted, const int32_t instance_num,
                      const int32_t elem_cnt_per_instance, const int32_t k, int32_t* fw_buf,
                      int32_t* indices_ptr, T* values_ptr) {
    std::vector<int32_t> indices_buf(elem_cnt_per_instance);
    FOR_RANGE(int32_t, i, 0, instance_num) {
      std::iota(indices_buf.begin(), indices_buf.end(), 0);
      std::nth_element(
          indices_buf.begin(), indices_buf.begin() + k - 1, indices_buf.end(),
          [&](const int32_t lhs, const int32_t rhs) { return in_ptr[lhs] > in_ptr[rhs]; });
      if (sorted) {
        std::sort(indices_buf.begin(), indices_buf.begin() + k - 1,
                  [&](const int32_t lhs, const int32_t rhs) { return in_ptr[lhs] > in_ptr[rhs]; });
      }
      FOR_RANGE(int32_t, j, 0, k) {
        if (indices_ptr) { indices_ptr[i * k + j] = indices_buf[j]; }
        if (values_ptr) {
          values_ptr[i * k + j] = in_ptr[i * elem_cnt_per_instance + indices_buf[j]];
        }
      }
    }
  }
};

#define INSTANTIATE_TOP_K_KERNEL_UTIL(type_cpp, type_proto) \
  template struct TopKKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_TOP_K_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kTopKConf, TopKKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
