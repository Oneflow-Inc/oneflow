#include <cstring>
#include "oneflow/core/kernel/rle_util.h"
#include "oneflow/core/common/util.h"

extern "C" {
#include "maskApi.h"
}

namespace oneflow {

namespace RleUtil {

size_t Encode(uint32_t* buf, const uint8_t* mask, size_t h, size_t w) {
  RLE rle;
  rleEncode(&rle, mask, h, w, 1);
  size_t len = rle.m;
  std::memcpy(buf, rle.cnts, len * sizeof(buf[0]));
  rleFree(&rle);
  return len;
}

void PolygonXy2ColMajorMask(const double* xy, size_t num_xy, size_t h, size_t w, uint8_t* mask) {
  CHECK_EQ(num_xy % 2, 0);
  RLE rle;
  rleFrPoly(&rle, xy, num_xy / 2, h, w);
  rleDecode(&rle, mask, 1);
  rleFree(&rle);
}

}  // namespace RleUtil

}  // namespace oneflow
