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
    out[i] = std::distance(values, std::max_element(values, values + instance_size));
  }
}

template<typename T>
void ForwardPartDataContentTopK(const T* in, const Range& range, const int32_t instance_size,
                                const int32_t k, const bool sorted, int32_t* indices,
                                int32_t* out) {
  CHECK_NOTNULL(indices);
  FOR_RANGE(int32_t, i, range.begin(), range.end()) {
    const int32_t offset = i * instance_size;
    int32_t* indices = indices + offset;
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
void CpuTopK(DeviceCtx* ctx, const T* in, int32_t* indices, int32_t instance_num,
             int32_t instance_size, int32_t k, bool sorted, int32_t* out) {
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
        ForwardPartDataContentTopK(in, range, instance_size, k, sorted, indices, out);
      }
      bc.Decrease();
    });
  }
  bc.WaitUntilCntEqualZero();
}

#define INSTANTIATE_CPU_TOP_K(T, type_proto)                                                     \
  template void CpuTopK<T>(DeviceCtx * ctx, const T* in, int32_t* indices, int32_t instance_num, \
                           int32_t instance_size, int32_t k, bool sorted, int32_t* out);
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CPU_TOP_K, ARITHMETIC_DATA_TYPE_SEQ)
#undef INSTANTIATE_CPU_TOP_K

template<DeviceType device_type, typename T>
void TopKKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  int32_t instance_size = in_blob->shape().dim_vec().back();
  int32_t instance_num = in_blob->shape().elem_cnt() / instance_size;
  int32_t k = this->op_conf().top_k_conf().k();
  const T* in = in_blob->dptr<T>();
  int32_t* out = out_blob->mut_dptr<int32_t>();

  if (this->op_conf().device_type() == DeviceType::kCPU) {
    CpuTopK(ctx.device_ctx, in, BnInOp2Blob("indices")->mut_dptr<int32_t>(), instance_num,
            instance_size, k, this->op_conf().top_k_conf().sorted(), out);
  } else if (this->op_conf().device_type() == DeviceType::kGPU) {
    if (instance_size <= 1024 || k == instance_size || k > 128) {
      GpuRadixSortTopK(ctx.device_ctx, in, BnInOp2Blob("indices")->mut_dptr<int32_t>(),
                       instance_num, instance_size, k,
                       BnInOp2Blob("temp_storage")->mut_dptr<void>(),
                       this->kernel_conf().top_k_conf().temp_storage_bytes(),
                       BnInOp2Blob("sorted_in")->mut_dptr<T>(),
                       BnInOp2Blob("sorted_indices")->mut_dptr<int32_t>(), out);
    } else {
      GpuHeapSelectionTopK(ctx.device_ctx, in, instance_num, instance_size, k, out);
    }
  } else {
    UNIMPLEMENTED();
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kTopKConf, TopKKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
