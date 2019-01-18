
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/top_k_kernel.h"

namespace oneflow {

template<typename T>
struct TopKKernelUtil<DeviceType::kGPU, T> {
  static void Forward(const T* prediction_ptr, const int64_t instance_num,
                      const int64_t elem_cnt_per_instance, const int64_t k, T* fw_buf,
                      T* indices_ptr, T* values_ptr) {
    UNIMPLEMENTED();
  }
};

#define INSTANTIATE_TOP_K_KERNEL_UTIL(type_cpp, type_proto) \
  template struct TopKKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_TOP_K_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow