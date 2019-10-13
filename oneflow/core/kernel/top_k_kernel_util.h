#ifndef ONEFLOW_CORE_KERNEL_TOP_K_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_TOP_K_KERNEL_UTIL_H_

namespace oneflow {

template<DeviceType devcie_type>
struct TopKKernelUtil {
  static int32_t ExtractBlobK(const Blob* k_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_TOP_K_KERNEL_UTIL_H_
