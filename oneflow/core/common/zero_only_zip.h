#ifndef ONEFLOW_CORE_COMMON_ZERO_ONLY_ZIP_H_
#define ONEFLOW_CORE_COMMON_ZERO_ONLY_ZIP_H_

#include <memory>
#include "oneflow/core/common/sized_buffer_view.h"

namespace oneflow {

struct ZeroOnlyZipUtil final {
  void ZipToSizedBuffer(const char* data, size_t size, SizedBufferView* sized_buffer);
  void UnzipToExpectedSize(const SizedBufferView& size_buffer, char* data, size_t expected_size);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ZERO_ONLY_ZIP_H_
