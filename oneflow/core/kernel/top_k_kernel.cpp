#include "oneflow/core/kernel/top_k_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void TopKKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction_blob = BnInOp2Blob("prediction");
  Blob* fw_buf_blob = BnInOp2Blob("fw_buf");
  Blob* indices_blob = BnInOp2Blob("indices");
  Blob* values_blob = BnInOp2Blob("values");
  const auto& conf = this->op_conf().top_k_conf();

  const int64_t elem_cnt_per_instance = prediction_blob->shape().dim_vec().back();
  const int64_t instance_num = prediction_blob->shape().elem_cnt() / elem_cnt_per_instance;

  T* indices_ptr = indices_blob ? indices_blob->mut_dptr<T>() : nullptr;
  T* values_ptr = values_blob ? values_blob->mut_dptr<T>() : nullptr;
  TopKKernelUtil<device_type, T>::Forward(prediction_blob->dptr<T>(), instance_num,
                                          elem_cnt_per_instance, conf.k(),
                                          fw_buf_blob->mut_dptr<T>(), indices_ptr, values_ptr);
}

template<typename T>
struct TopKKernelUtil<DeviceType::kCPU, T> {
  static void Forward(const T* prediction_ptr, const int64_t instance_num,
                      const int64_t elem_cnt_per_instance, const int64_t k, T* fw_buf,
                      T* indices_ptr, T* values_ptr) {
    std::vector<int64_t> indices_buf(elem_cnt_per_instance);
    FOR_RANGE(int64_t, i, 0, instance_num) {
      std::iota(indices_buf.begin(), indices_buf.end(), 0);
      std::nth_element(indices_buf.begin(), indices_buf.begin() + k - 1, indices_buf.end(),
                       [&](const int64_t lhs, const int64_t rhs) {
                         return indices_buf[lhs] >= indices_buf[rhs];
                       });
      FOR_RANGE(int64_t, j, 0, k) {
        if (!indices_ptr) { indices_ptr[i * k + j] = indices_buf.at(j); }
        if (!values_ptr) {
          values_ptr[i * k + j] = prediction_ptr[i * elem_cnt_per_instance + indices_buf.at(j)];
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
