#ifndef ONEFLOW_CORE_KERNEL_RLE_H_
#define ONEFLOW_CORE_KERNEL_RLE_H_

#include <cstddef>
#include <cstdint>

namespace oneflow {

namespace RleUtil {

size_t Encode(uint32_t* buf, const uint8_t* mask, size_t h, size_t w);
void PolygonXy2ColMajorMask(const double* xy, size_t num_xy, size_t h, size_t w, uint8_t* mask);

}  // namespace RleUtil

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RLE_H_
