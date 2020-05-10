#include "oneflow/customized/data/dataset.h"

namespace oneflow {

template<>
void PrepareEmptyTensor(TensorBuffer& tensor, int32_t tensor_init_bytes) {
  tensor.reset();
  // Initialize tensors to a set size to limit expensive reallocations
  tensor.Resize({tensor_init_bytes}, DataType::kChar);
}

}  // namespace oneflow
