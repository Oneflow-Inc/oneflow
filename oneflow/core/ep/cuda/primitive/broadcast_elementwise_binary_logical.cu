#include "oneflow/core/ep/cuda/primitive/broadcast_elementwise_binary.cuh"

namespace oneflow {

namespace ep {
namespace primitive {

#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_LOGICAL_ENTRY(      \
    binary_op, src_data_type_pair, dst_data_type_pair)                            \
    template std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(src_data_type_pair), OF_PP_PAIR_FIRST(dst_data_type_pair)>(); 

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_LOGICAL_ENTRY,
    BINARY_COMPARISION_OP_SEQ BINARY_LOGICAL_OP_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ,
    CUDA_PRIMITIVE_INT8_TYPE_SEQ);

//make ->inst
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
