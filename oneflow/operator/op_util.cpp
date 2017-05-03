#include "operator/op_util.h"

namespace oneflow {

std::pair<uint32_t, uint32_t> CheckDimPara4CnnOrPooling(uint32_t val,
                                                        uint32_t val_h,
                                                        uint32_t val_w) {
  CHECK(val == 0 || (val_h == 0 && val_w == 0)
                 || (val == val_h && val == val_w))
    << "don't set parameter both 'val' and ('val_h','val_w')"
    << " in convolution_conf or pooling_conf ";
  if (val != 0) {
    return {val, val};
  }
  return {val_h, val_w};
}

}  // namespace oneflow
