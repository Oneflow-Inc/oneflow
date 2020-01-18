#ifndef ONEFLOW_CORE_COMMON_ZERO_ONLY_ZIP_H_
#define ONEFLOW_CORE_COMMON_ZERO_ONLY_ZIP_H_

#include <memory>
#include "oneflow/core/common/sized_buffer_view.h"

namespace oneflow {

struct ZeroOnlyZipUtil final {
  void ZipToSizedBuffer(const char* data, size_t size, SizedBufferView* sized_buffer) {
    size_t cur_index = 0;
    size_t former_index = 0;
    size_t count = 0;
    size_t cursor = 0;
    while (cur_index < size) {
      if (data[cur_index] != 0x00 && cur_index != size) {
        cur_index++;
        count++;
      }
      else {
        if (count != 0){
          size_buffer -> data[cursor++] = count;
          for (; former_index < cur_index; former_index++) {
            size_buffer -> data[cursor++] = data[former_index];
          }
        }
        count = 0;
        if (cur_index == size) break;
        while (cur_index < size && data[cur_index] == 0x00) {
          count++;
          cur_index++;
        }
        former_index = cur_index;
        sized_buffer -> data[cursor++] = -count;
        count = 0
      }
    }
  }

  void UnzipToExpectedSize(const SizedBufferView& size_buffer, char* data, size_t expected_size) {
    size_t cursor = 0;
    size_t cur_index = 0;
    while (cur_index < expected_size) {
      if (int(size_buffer.data[cursor]) > 0) {
        int cur_size = size_buffer.data[cursor++];
        for (int index = 0; index < cur_size; index++, cur_index++) {
          *(data + cur_index) = size_buffer.data[cursor++];
        }
      }
      else {
        int cur_size = -int(size_buffer.data[cursor++];
        for (int index = 0; index < cur_size; index++, cur_index++) {
          *(data + cur_index) = 0x00;
        }
      }
    }
  }
};

}

#endif  // ONEFLOW_CORE_COMMON_ZERO_ONLY_ZIP_H_
