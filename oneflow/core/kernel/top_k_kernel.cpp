#include "oneflow/core/kernel/top_k_kernel.h"

namespace oneflow {

template<typename T>
void TopKKernel<T>::ForwardDataContent(const KernelCtx& ctx,
                                       std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* fw_buf_blob = BnInOp2Blob("fw_buf");
  Blob* out_blob = BnInOp2Blob("out");

  CHECK_LE(in_blob->shape().elem_cnt(), GetMaxVal<int32_t>());
  const int32_t instance_size = static_cast<int32_t>(in_blob->shape().dim_vec().back());
  const int32_t instance_num = static_cast<int32_t>(in_blob->shape().elem_cnt() / instance_size);
  const T* in = in_blob->dptr<T>();
  int32_t* fw_buf = fw_buf_blob->mut_dptr<int32_t>();
  int32_t* out = out_blob->mut_dptr<int32_t>();
  const auto& conf = this->op_conf().top_k_conf();
  const int32_t k = conf.k();
  FOR_RANGE(int32_t, i, 0, instance_num) {
    std::iota(fw_buf, fw_buf + instance_size, 0);
    const int32_t offset = i * instance_size;
    auto comp = [&](const int32_t lhs, const int32_t rhs) {
      const T l = in[offset + lhs];
      const T r = in[offset + rhs];
      if (l == r) {
        return lhs < rhs;
      } else {
        return l > r;
      }
    };
    std::nth_element(fw_buf, fw_buf + k, fw_buf + instance_size, comp);
    if (k > 1 && conf.sorted()) { std::sort(fw_buf, fw_buf + k, comp); }
    std::copy(fw_buf, fw_buf + k, out + i * k);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kTopKConf, TopKKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
