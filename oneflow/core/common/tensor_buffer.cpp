#include "oneflow/core/common/tensor_buffer.h"

namespace oneflow {

double TensorBuffer::growth_factor_ = 1.0;
double TensorBuffer::shrink_threshold_ = 0.9;

}  // namespace oneflow
