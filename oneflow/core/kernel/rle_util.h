#ifndef ONEFLOW_CORE_KERNEL_RLE_H_
#define ONEFLOW_CORE_KERNEL_RLE_H_

#include <cstddef>
#include <cstdint>

namespace oneflow {

size_t RleEncode(uint8_t* buf, const uint8_t* mask, size_t h, size_t w);
}

#endif  // ONEFLOW_CORE_KERNEL_RLE_H_
