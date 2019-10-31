#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

template<typename T>
void ForwardPartDataContentTopOne(const T* in_ptr, const Range& range, const int32_t instance_size,
                                  int32_t* out_ptr) {
  FOR_RANGE(int32_t, i, range.begin(), range.end()) {
    const T* in_ptr_i = in_ptr + i * instance_size;
    out_ptr[i] = std::distance(in_ptr_i, std::max_element(in_ptr_i, in_ptr_i + instance_size));
  }
}

template<typename T>
void ForwardPartDataContentTopK(const T* in_ptr, int32_t* indices_ptr, const Range& range,
                                const int32_t instance_size, const int32_t k, const bool sorted,
                                int32_t* out_ptr) {
  CHECK_NOTNULL(indices_ptr);
  FOR_RANGE(int32_t, i, range.begin(), range.end()) {
    const int32_t offset = i * instance_size;
    int32_t* indices_ptr_i = indices_ptr + offset;
    const T* in_ptr_i = in_ptr + offset;
    std::iota(indices_ptr_i, indices_ptr_i + instance_size, 0);
    auto comp = [&](const int32_t lhs, const int32_t rhs) {
      const T l = in_ptr_i[lhs];
      const T r = in_ptr_i[rhs];
      if (l == r) {
        return lhs < rhs;
      } else {
        return l > r;
      }
    };
    std::nth_element(indices_ptr_i, indices_ptr_i + k, indices_ptr_i + instance_size, comp);
    if (sorted) { std::sort(indices_ptr_i, indices_ptr_i + k, comp); }
    std::copy(indices_ptr_i, indices_ptr_i + k, out_ptr + i * k);
  }
}

}  // namespace

template<typename T>
void GpuHeapSelectionTopK(DeviceCtx* ctx, const T* in_ptr, int32_t instance_num,
                          int32_t instance_size, int32_t k, int32_t* out_ptr);

template<typename T>
void GpuRadixSortTopK(DeviceCtx* ctx, const T* in_ptr, int32_t* indices_ptr, int32_t instance_num,
                      int32_t instance_size, int32_t k, void* temp_storage_ptr,
                      size_t temp_storage_bytes, T* sorted_in_ptr, int32_t* sorted_indices_ptr,
                      int32_t* out_ptr);

template<typename T>
void CpuTopK(DeviceCtx* ctx, const T* in_ptr, int32_t* indices_ptr, int32_t instance_num,
             int32_t instance_size, int32_t k, bool sorted, int32_t* out_ptr) {
  const int32_t part_num =
      std::min(instance_num, Global<ThreadMgr>::Get()->compute_thread_pool()->thread_num());
  const BalancedSplitter bs(instance_num, part_num);
  BlockingCounter bc(part_num);
  FOR_RANGE(int32_t, part_id, 0, part_num) {
    const Range range = bs.At(part_id);
    Global<ThreadMgr>::Get()->compute_thread_pool()->AddWork([=, &bc]() {
      if (k == 1) {
        ForwardPartDataContentTopOne(in_ptr, range, instance_size, out_ptr);
      } else {
        ForwardPartDataContentTopK(in_ptr, indices_ptr, range, instance_size, k, sorted, out_ptr);
      }
      bc.Decrease();
    });
  }
  bc.WaitUntilCntEqualZero();
}

template<DeviceType device_type, typename T>
class TopKKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TopKKernel);
  TopKKernel() = default;
  ~TopKKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    int32_t instance_size = in_blob->shape().At(in_blob->shape().NumAxes() - 1);
    int32_t instance_num = in_blob->shape().elem_cnt() / instance_size;
    // temp solution: we allow k > instance_size
    // CHECK_LE(k, instance_size)
    int32_t k = std::min(static_cast<int32_t>(this->op_conf().top_k_conf().k()), instance_size);
    const T* in_ptr = in_blob->dptr<T>();
    int32_t* out_ptr = out_blob->mut_dptr<int32_t>();

    if (this->op_conf().device_type() == DeviceType::kCPU) {
      Blob* indices_blob = BnInOp2Blob("indices");
      int32_t* indices_ptr = indices_blob ? indices_blob->mut_dptr<int32_t>() : nullptr;
      CpuTopK(ctx.device_ctx, in_ptr, indices_ptr, instance_num, instance_size, k,
              this->op_conf().top_k_conf().sorted(), out_ptr);
    } else if (this->op_conf().device_type() == DeviceType::kGPU) {
      if (k > 128) {
        GpuRadixSortTopK(ctx.device_ctx, in_ptr, BnInOp2Blob("indices")->mut_dptr<int32_t>(),
                         instance_num, instance_size, k,
                         BnInOp2Blob("temp_storage")->mut_dptr<void>(),
                         this->kernel_conf().top_k_conf().temp_storage_bytes(),
                         BnInOp2Blob("sorted_in")->mut_dptr<T>(),
                         BnInOp2Blob("sorted_indices")->mut_dptr<int32_t>(), out_ptr);
      } else {
        GpuHeapSelectionTopK(ctx.device_ctx, in_ptr, instance_num, instance_size, k, out_ptr);
      }
    } else {
      UNIMPLEMENTED();
    }
  }
};

#define REGISTER_TOP_K_KERNEL(dtype)                                                      \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kTopKConf, DeviceType::kCPU, dtype, \
                                        TopKKernel<DeviceType::kCPU, dtype>)              \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kTopKConf, DeviceType::kGPU, dtype, \
                                        TopKKernel<DeviceType::kGPU, dtype>)

REGISTER_TOP_K_KERNEL(float);
REGISTER_TOP_K_KERNEL(double);
REGISTER_TOP_K_KERNEL(int8_t);
REGISTER_TOP_K_KERNEL(int32_t);
REGISTER_TOP_K_KERNEL(int64_t);

}  // namespace oneflow
