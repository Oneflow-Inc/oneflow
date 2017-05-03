#ifndef ONEFLOW_OPERATOR_OP_UTIL_H_
#define ONEFLOW_OPERATOR_OP_UTIL_H_

#include <stdint.h>
#include <string>
#include <utility>
#include <functional>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "common/shape.h"
#include "common/util.h"

namespace oneflow {

// check the parameter which user set in convolution layer and pooling layer
// is legal and return h and w
// Param:
//  val = [pad | kernel | stride]
//  val_h = [pad_h | kernel_h | stride_h]
//  val_w = [pad_w | kernel_w | stride_w]
// Return:
//  (val_h, val_w)
// if user both set the val and (val_h, val_w), a LOG(FATAL) will be called
// else return the real val_h and val_w
std::pair<uint32_t, uint32_t> CheckDimPara4CnnOrPooling(uint32_t val,
                                                        uint32_t val_h,
                                                        uint32_t val_w);

}  // namespace oneflow

#endif  // ONEFLOW_OPERATOR_OP_UTIL_H_
