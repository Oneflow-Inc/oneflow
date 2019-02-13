#include "oneflow/core/kernel/top_k_kernel.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

namespace {

template<typename T>
void ForwardPartDataContentTopOne(const T* in, const Range& range, const int32_t instance_size,
                                  int32_t* out) {
  FOR_RANGE(int32_t, i, range.begin(), range.end()) {
    const T* values = in + i * instance_size;
    T max_val = values[0];
    int32_t max_idx = 0;
    FOR_RANGE(int32_t, j, 0, instance_size) {
      if (values[j] > max_val) {
        max_val = values[j];
        max_idx = j;
      }
    }
    out[i] = max_idx;
  }
}

template<typename T>
void ForwardPartDataContentTopK(const T* in, const Range& range, const int32_t instance_size,
                                const int32_t k, const bool sorted, int32_t* fw_buf, int32_t* out) {
  CHECK_NOTNULL(fw_buf);
  FOR_RANGE(int32_t, i, range.begin(), range.end()) {
    const int32_t offset = i * instance_size;
    int32_t* indices = fw_buf + offset;
    const T* values = in + offset;
    std::iota(indices, indices + instance_size, 0);
    auto comp = [&](const int32_t lhs, const int32_t rhs) {
      const T l = values[lhs];
      const T r = values[rhs];
      if (l == r) {
        return lhs < rhs;
      } else {
        return l > r;
      }
    };
    std::nth_element(indices, indices + k, indices + instance_size, comp);
    if (sorted) { std::sort(indices, indices + k, comp); }
    std::copy(indices, indices + k, out + i * k);
  }
}

}  // namespace

template<typename T>
struct TopKKernelUtil<DeviceType::kCPU, T> {
  static void Forward(DeviceCtx* ctx, const T* in, const int32_t instance_num,
                      const int32_t instance_size, const int32_t k, const bool sorted,
                      int32_t* fw_buf, int32_t* out) {
    const int32_t part_num =
        std::min(instance_num, Global<ThreadMgr>::Get()->compute_thread_pool()->thread_num());
    const BalancedSplitter bs(instance_num, part_num);
    BlockingCounter bc(part_num);
    FOR_RANGE(int32_t, part_id, 0, part_num) {
      const Range range = bs.At(part_id);
      Global<ThreadMgr>::Get()->compute_thread_pool()->AddWork([=, &bc]() {
        if (k == 1) {
          ForwardPartDataContentTopOne(in, range, instance_size, out);
        } else {
          ForwardPartDataContentTopK(in, range, instance_size, k, sorted, fw_buf, out);
        }
        bc.Decrease();
      });
    }
    bc.WaitUntilCntEqualZero();
  }
};

template<DeviceType device_type, typename T>
void TopKKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* fw_buf_blob = BnInOp2Blob("fw_buf");
  Blob* out_blob = BnInOp2Blob("out");

  CHECK_LE(in_blob->shape().elem_cnt(), GetMaxVal<int32_t>());
  const int32_t instance_size = static_cast<int32_t>(in_blob->shape().dim_vec().back());
  const int32_t instance_num = static_cast<int32_t>(in_blob->shape().elem_cnt() / instance_size);
  const T* in = in_blob->dptr<T>();
  int32_t* fw_buf = fw_buf_blob ? fw_buf_blob->mut_dptr<int32_t>() : nullptr;
  int32_t* out = out_blob->mut_dptr<int32_t>();
  const auto& conf = this->op_conf().top_k_conf();
  TopKKernelUtil<device_type, T>::Forward(ctx.device_ctx, in, instance_num, instance_size, conf.k(),
                                          conf.sorted(), fw_buf, out);
}

#define INSTANTIATE_TOP_K_KERNEL_UTIL(type_cpp, type_proto) \
  template struct TopKKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_TOP_K_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)
#undef INSTANTIATE_TOP_K_KERNEL_UTIL

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kTopKConf, TopKKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
