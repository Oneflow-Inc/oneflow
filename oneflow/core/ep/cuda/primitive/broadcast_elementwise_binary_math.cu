#include "oneflow/core/ep/cuda/primitive/broadcast_elementwise_binary.cuh"

namespace oneflow {

namespace ep {
namespace primitive {

#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY(binary_op, data_type_pair) \
    template std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(data_type_pair)>();

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY,
                                             BINARY_MATH_OP_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ);

//make ->inst
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
