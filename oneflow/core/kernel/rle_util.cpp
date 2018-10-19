#include "oneflow/core/kernel/rle_util.h"

extern "C" {

#include <maskApi.h>
}

namespace oneflow {

size_t RleEncode(uint8_t* buf, const uint8_t* mask, size_t h, size_t w) { return 0; }

}  // namespace oneflow
