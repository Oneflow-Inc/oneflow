#include <memory>
#include "oneflow/core/common/sized_buffer_view.h"

namespace oneflow {

struct ZeroOnlyZipUtil final {
  void ZipToSizedBuffer(const char* data, size_t size, SizedBufferView* sized_buffer) {
    size_t cur_index = 0, count = 0, cursor = 0;
    while (cur_index < size) {
      count++;
      if (data[cur_index + count - 1] != 0x00
          && (data[cur_index + count] == 0x00 || (cur_index + count) == size)) {
        size_buffer->data[cursor++] = count;
        memcpy(&(size_buffer->data[cursor]), &data[cur_index], count);
        cursor += count;
        cur_index += count;
        count = 0;
      } else if (data[cur_index + count - 1] == 0x00
                 && (data[cur_index + count] != 0x00 || (cur_index + count) == size)) {
        size_buffer->data[cursor++] = -count;
        cur_index += count;
        count = 0;
      }
    }
  }

  void UnzipToExpectedSize(const SizedBufferView& size_buffer, char* data, size_t expected_size) {
    size_t cursor = 0, cur_index = 0;
    while (cur_index < expected_size) {
      if (int(size_buffer.data[cursor]) > 0) {
        int cur_size = size_buffer.data[cursor++];
        memcpy(data + cur_index, &(size_buffer.data[cursor]), cur_size);
        cursor += cur_size;
        cur_index += cur_size;
      } else {
        int cur_size = -int(size_buffer.data[cursor++];
        for (int index = 0; index < cur_size; index++, cur_index++) {
          *(data + cur_index) = 0x00;
        }
      }
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ZERO_ONLY_ZIP_H_
