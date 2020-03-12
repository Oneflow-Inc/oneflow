#include "oneflow/customized/kernels/clip_by_value_kernel.h"

namespace oneflow {

template<typename T>
struct DeviceClip<DeviceType::kCPU, T> {
  OF_DEVICE_FUNC static T Min(const T value, const T min_value) {
    return std::min(value, min_value);
  }

  OF_DEVICE_FUNC static T Max(const T value, const T max_value) {
    return std::max(value, max_value);
  }
};

template<typename T>
struct ClipValuesUtil<DeviceType::kCPU, T> {
  static void ByMin(DeviceCtx* ctx, int64_t num_values, const T* values, const T* min_value,
                    T* out_ptr) {
    using namespace std;
    ClipValuesByMin<DeviceType::kCPU>(num_values, values, *min_value, out_ptr);
  }

  static void ByMax(DeviceCtx* ctx, int64_t num_values, const T* values, const T* max_value,
                    T* out_ptr) {
    using namespace std;
    ClipValuesByMin<DeviceType::kCPU>(num_values, values, *max_value, out_ptr);
  }

  static void ByMinMax(DeviceCtx* ctx, int64_t num_values, const T* values, const T* min_value,
                       const T* max_value, T* out_ptr) {
    using namespace std;
    ClipValuesByMinMax<DeviceType::kCPU>(num_values, values, *min_value, *max_value, out_ptr);
  }
};

template<typename T>
struct ClipGradUtil<DeviceType::kCPU, T> {
  static void ByMin(DeviceCtx* ctx, int64_t num_values, const T* values, const T* min_value,
                    T* grad_ptr) {
    ClipGradByMin<DeviceType::kCPU>(num_values, values, *min_value, grad_ptr);
  }

  static void ByMax(DeviceCtx* ctx, int64_t num_values, const T* values, const T* max_value,
                    T* grad_ptr) {
    ClipGradByMin<DeviceType::kCPU>(num_values, values, *max_value, grad_ptr);
  }

  static void ByMinMax(DeviceCtx* ctx, int64_t num_values, const T* values, const T* min_value,
                       const T* max_value, T* grad_ptr) {
    ClipGradByMinMax<DeviceType::kCPU>(num_values, values, *min_value, *max_value, grad_ptr);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CLIP_UTIL, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CLIP_KERNELS, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
