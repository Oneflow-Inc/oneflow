#include "oneflow/core/kernel/top_k_kernel.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/thread/thread_manager.h"

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

template<DeviceType device_type, typename T>
void TopKKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  CHECK(in_blob->has_dim0_valid_num_field());
  const int32_t dim0_valid_num = in_blob->dim0_valid_num(0);
  int32_t k = this->op_conf().top_k_conf().k();
  if (device_type == DeviceType::kCPU) {
    if (k > 1) { BnInOp2Blob("indices")->set_dim0_valid_num(0, dim0_valid_num); }
  } else if (device_type == DeviceType::kGPU) {
    if (k > 128) {
      BnInOp2Blob("indices")->set_dim0_valid_num(0, dim0_valid_num);
      BnInOp2Blob("sorted_in")->set_dim0_valid_num(0, dim0_valid_num);
      BnInOp2Blob("sorted_indices")->set_dim0_valid_num(0, dim0_valid_num);
    }
  } else {
    UNIMPLEMENTED();
  }
  BnInOp2Blob("out")->set_dim0_valid_num(0, dim0_valid_num);
}

template<DeviceType device_type, typename T>
void TopKKernel<device_type, T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  CHECK(in_blob->has_instance_shape_field());
  std::vector<int64_t> dim_vec = in_blob->instance_shape().dim_vec();
  dim_vec.back() = this->op_conf().top_k_conf().k();
  const Shape instance_shape = Shape(dim_vec);
  int32_t k = this->op_conf().top_k_conf().k();
  if (device_type == DeviceType::kCPU) {
    if (k > 1) { BnInOp2Blob("indices")->set_instance_shape(instance_shape); }
  } else if (device_type == DeviceType::kGPU) {
    if (k > 128) {
      BnInOp2Blob("indices")->set_instance_shape(instance_shape);
      BnInOp2Blob("sorted_in")->set_instance_shape(instance_shape);
      BnInOp2Blob("sorted_indices")->set_instance_shape(instance_shape);
    }
  } else {
    UNIMPLEMENTED();
  }
  BnInOp2Blob("out")->set_instance_shape(instance_shape);
}

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
void TopKKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  int32_t instance_size = in_blob->shape().dim_vec().back();
  int32_t instance_num = in_blob->shape().elem_cnt() / instance_size;
  int32_t k = this->op_conf().top_k_conf().k();
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

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kTopKConf, TopKKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
