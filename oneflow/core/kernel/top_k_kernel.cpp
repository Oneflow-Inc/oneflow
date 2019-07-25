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
void CpuTopK(DeviceCtx* ctx, const T* in, const int32_t instance_num, const int32_t instance_size,
             const int32_t k, const bool sorted, int32_t* indices, int32_t* out) {
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

#define INSTANTIATE_CPU_TOP_K(T, type_proto)                                                \
  template void CpuTopK<T>(DeviceCtx * ctx, const T* in, const int32_t instance_num,        \
                           const int32_t instance_size, const int32_t k, const bool sorted, \
                           int32_t* indices, int32_t* out);
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CPU_TOP_K, ARITHMETIC_DATA_TYPE_SEQ)
#undef INSTANTIATE_CPU_TOP_K

template<DeviceType device_type, typename T>
void TopKKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");

  CHECK_LE(in_blob->shape().elem_cnt(), GetMaxVal<int32_t>());
  int32_t instance_size = static_cast<int32_t>(in_blob->shape().dim_vec().back());
  int32_t instance_num = static_cast<int32_t>(in_blob->shape().elem_cnt() / instance_size);
  const T* in = in_blob->dptr<T>();
  int32_t* out = out_blob->mut_dptr<int32_t>();
  auto& top_k_op_conf = this->op_conf().top_k_conf();

  if (this->op_conf().device_type() == DeviceType::kCPU) {
    Blob* cpu_indices_blob = BnInOp2Blob("cpu_indices");
    CHECK_NOTNULL(cpu_indices_blob);
    int32_t* cpu_indices = cpu_indices_blob->mut_dptr<int32_t>();
    CpuTopK<T>(ctx.device_ctx, in, instance_num, instance_size, top_k_op_conf.k(),
               top_k_op_conf.sorted(), cpu_indices, out);
  } else if (this->op_conf().device_type() == DeviceType::kGPU) {
    if (instance_size <= 1000 || top_k_op_conf.k() == instance_size || top_k_op_conf.k() > 512) {
      Blob* gpu_indices_blob = BnInOp2Blob("gpu_indices");
      CHECK_NOTNULL(gpu_indices_blob);
      Blob* sorted_in_blob = BnInOp2Blob("sorted_in");
      CHECK_NOTNULL(sorted_in_blob);
      Blob* sorted_indices_blob = BnInOp2Blob("sorted_indices");
      CHECK_NOTNULL(sorted_indices_blob);
      Blob* temp_storage_blob = BnInOp2Blob("temp_storage");
      CHECK_NOTNULL(temp_storage_blob);

      int32_t* gpu_indices = gpu_indices_blob->mut_dptr<int32_t>();
      T* sorted_in = sorted_in_blob->mut_dptr<T>();
      int32_t* sorted_indices = sorted_indices_blob->mut_dptr<int32_t>();
      void* temp_storage = temp_storage_blob->mut_dptr<void>();
      GpuRadixSortTopK(ctx.device_ctx, in, instance_num, instance_size, top_k_op_conf.k(),
                       gpu_indices, sorted_in, sorted_indices, temp_storage,
                       this->kernel_conf().top_k_conf().temp_storage_bytes(), out);
    } else {
      GpuHeapSelectionTopK<T>(ctx.device_ctx, in, instance_num, instance_size, top_k_op_conf.k(),
                              out);
    }
  } else {
    UNIMPLEMENTED();
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kTopKConf, TopKKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
