#ifndef ONEFLOW_CORE_LIB_CRC32C_H_
#define ONEFLOW_CORE_LIB_CRC32C_H_

#include <stdint.h>
#include <cstddef>

namespace oneflow{
namespace crc32c{

uint32_t Mask(uint32_t crc);
uint32_t Unmask(uint32_t masked_crc);
inline uint32_t DecodeFixed32(const char* ptr);
uint32_t Unmask(uint32_t masked_crc);
uint32_t Extend(uint32_t crc, const char *buf, size_t size);
bool IsLittleEndian();

} // namespace crc32c
} // namespace oneflow

#endif