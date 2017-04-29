#include <stdint.h>
#include <utility>
#include "glog/logging.h"

namespace oneflow {


// check the parameter which user set in convolution layer and pooling layer
// is legal and return h and w
// Param:
//  val = {pad, kernel, stride}
//  val_h = {pad_h, kernel_h, stride_h}
//  val_w = {pad_w, kernel_w, stride_w}
// Return:
//  (val_h, val_w)
// if user both set the val and (val_h, val_w), a LOG(FATAL) will be called
// else return the real val_h and val_w
std::pair<uint32_t, uint32_t> CheckDimPara(uint32_t val, uint32_t val_h, 
                                           uint32_t val_w) {
  if (val != 0 && (val_h != 0 || val_w != 0) && (val != val_h || val != val_w)) {
    LOG(FATAL) << "don't set parameter both 'val' and ('val_h','val_w')"
      << " in convolution_conf or pooling_conf ";
  }
  if (val != 0) {
    return std::make_pair(val, val);
  }
  return std::make_pair(val_h, val_w);
}

} // namespace oneflow