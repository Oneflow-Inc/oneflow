#include "oneflow/core/primitive/include/unary_op.h"
#include "oneflow/core/primitive/cuda/type_seq.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/stream/cuda_stream_context.h"

namespace oneflow{

namespace primitive{

template<typename Out, typename In>
struct UnaryFunctor<DeviceType::kGPU, UnaryOpList::kIdentity, Out, In>{
    __device__ Out operator()(In src) const {
        return static_cast<Out>(src); 
    }
}; 

namespace{

template<UnaryOpList unary_enum, typename Out, typename In>
class UnaryOpImpl: public UnaryOp{
public: 
    OF_DISALLOW_COPY_AND_MOVE(UnaryOpImpl); 
    UnaryOpImpl() = default;
    ~UnaryOpImpl() override = default; 

    void Launch(StreamContext* stream_ctx, size_t count, void* dst_ptr, const void* src_ptr) override {
        auto* cuda_stream_ctx = CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx));
        OF_CUDA_CHECK((cuda::elementwise::Unary<UnaryFunctor<DeviceType::kGPU, unary_enum, Out, In>, Out, In>(
            UnaryFunctor<DeviceType::kGPU, unary_enum, Out, In>(), count, reinterpret_cast<Out*>(dst_ptr),
            reinterpret_cast<const In*>(src_ptr), cuda_stream_ctx->cuda_stream())));
      }

}; 

template<UnaryOpList unary_enum, typename Out, typename In>
std::unique_ptr<UnaryOp> NewUnaryOp() {
  return std::unique_ptr<UnaryOp>(new UnaryOpImpl<unary_enum, Out, In>());
}

class UnaryOpFactoryImpl : public UnaryOpFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnaryOpFactoryImpl);
  UnaryOpFactoryImpl() = default;
  ~UnaryOpFactoryImpl() override = default;

    std::unique_ptr<UnaryOp> New(UnaryOpList op_enum, DataType out, DataType in) override {
#define MAKE_NEW_UNARYOP_ENTRY(op_pair, out_pair, in_pair)                              \
      {std::make_tuple(OF_PP_PAIR_SECOND(op_pair), OF_PP_PAIR_SECOND(out_pair), OF_PP_PAIR_SECOND(in_pair)), \
        NewUnaryOp<OF_PP_PAIR_SECOND(op_pair), OF_PP_PAIR_FIRST(out_pair), OF_PP_PAIR_FIRST(in_pair)>},
    
        static const std::map<std::tuple<UnaryOpList, DataType, DataType>, std::function<std::unique_ptr<UnaryOp>()>>
            new_unary_op_handle{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                MAKE_NEW_UNARYOP_ENTRY, CPU_PRIMITIVE_UNARY_OP_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ)};
#undef MAKE_NEW_UNARYOP_ENTRY
    const auto it = new_unary_op_handle.find(std::make_tuple(op_enum, out, in));
    if (it != new_unary_op_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, UnaryOpFactory, UnaryOpFactoryImpl);

} // namespace 
} // namespace primitive
} // namespace oneflow