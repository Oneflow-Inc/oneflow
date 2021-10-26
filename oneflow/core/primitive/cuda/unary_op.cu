#include "oneflow/core/primitive/include/unary_op.h"
#include "oneflow/core/primitive/cuda/type_seq.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/stream/cuda_stream_context.h"

namespace oneflow{

namespace primitive{

template<DeviceType device, UnaryOpList unary_enum, typename Out, typename In, typename = void>
  struct UnaryFunctor; 

template<typename Out, typename In>
struct UnaryFunctor<DeviceType::kGPU, UnaryOpList::kRelu, Out, In, void>{
    __device__ Out operator()(In src) const {
      return src ? src > static_cast<In>(0) : static_cast<Out>(0); 
    }
}; 

template<typename In>
struct UnaryFunctor<DeviceType::kGPU, UnaryOpList::kRelu, half, In, typename std::enable_if<!std::is_same<In, half>::value>::type>{
    __device__ half operator()(In src) const {
      half src_tmp = static_cast<half>(src); 
      half zero_tmp = static_cast<half>(0.0); 
      if(__hlt(src_tmp, zero_tmp)){
        return zero_tmp; 
      }else{
        return src_tmp; 
      }
    }
}; 

#if CUDA_VERSION >= 11000

template<typename In>
struct UnaryFunctor<DeviceType::kGPU, UnaryOpList::kRelu, nv_bfloat16, In, typename std::enable_if<!(std::is_same<In, nv_bfloat16>::value || std::is_same<In, half>::value)>::type> {
  __device__ nv_bfloat16 operator()(In src) const {
    In zero_tmp = static_cast<In>(0.0); 
    return static_cast<nv_bfloat16>(static_cast<float>(src)) ? src > zero_tmp : static_cast<nv_bfloat16>(0.0); 
  }
};

template<typename Out>
struct UnaryFunctor<DeviceType::kGPU, UnaryOpList::kRelu, Out, nv_bfloat16, typename std::enable_if<!(std::is_same<Out, nv_bfloat16>::value || std::is_same<Out, half>::value)>::type> {
  __device__ Out operator()(nv_bfloat16 src) const {
    float src_tmp = static_cast<float>(src); 
    float zero_tmp = static_cast<float>(0.0); 
    return static_cast<Out>(src_tmp) ? src_tmp > zero_tmp : static_cast<Out>(0.0); 
  }
};

#endif  // CUDA_VERSION >= 11000

namespace{

template<UnaryOpList unary_enum, typename Out, typename In>
class ElementwiseUnaryOpImpl: public ElementwiseUnaryOp{
public: 
    OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryOpImpl); 
    ElementwiseUnaryOpImpl() = default;
    ~ElementwiseUnaryOpImpl() override = default; 

    void Launch(StreamContext* stream_ctx, size_t count, void* dst_ptr, const void* src_ptr) override {
        auto* cuda_stream_ctx = CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx));
        OF_CUDA_CHECK((cuda::elementwise::Unary<UnaryFunctor<DeviceType::kGPU, unary_enum, Out, In>, Out, In>(
            UnaryFunctor<DeviceType::kGPU, unary_enum, Out, In>(), count, reinterpret_cast<Out*>(dst_ptr),
            reinterpret_cast<const In*>(src_ptr), cuda_stream_ctx->cuda_stream())));
      }

}; 

template<UnaryOpList unary_enum, typename Out, typename In>
std::unique_ptr<ElementwiseUnaryOp> NewElementwiseUnaryOp() {
  return std::unique_ptr<ElementwiseUnaryOp>(new ElementwiseUnaryOpImpl<unary_enum, Out, In>());
}

class ElementwiseUnaryOpFactoryImpl : public ElementwiseUnaryOpFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryOpFactoryImpl);
  ElementwiseUnaryOpFactoryImpl() = default;
  ~ElementwiseUnaryOpFactoryImpl() override = default;

    std::unique_ptr<ElementwiseUnaryOp> New(UnaryOpList op_enum, DataType out, DataType in) override {
#define MAKE_NEW_UNARYOP_ENTRY(op_pair, out_pair, in_pair)                              \
      {std::make_tuple(OF_PP_PAIR_SECOND(op_pair), OF_PP_PAIR_SECOND(out_pair), OF_PP_PAIR_SECOND(in_pair)), \
        NewElementwiseUnaryOp<OF_PP_PAIR_SECOND(op_pair), OF_PP_PAIR_FIRST(out_pair), OF_PP_PAIR_FIRST(in_pair)>},
    
        static const std::map<std::tuple<UnaryOpList, DataType, DataType>, std::function<std::unique_ptr<ElementwiseUnaryOp>()>>
            new_elementwise_unary_op_handle{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                MAKE_NEW_UNARYOP_ENTRY, CPU_PRIMITIVE_UNARY_OP_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ)};
#undef MAKE_NEW_UNARYOP_ENTRY
    const auto it = new_elementwise_unary_op_handle.find(std::make_tuple(op_enum, out, in));
    if (it != new_elementwise_unary_op_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, ElementwiseUnaryOpFactory, ElementwiseUnaryOpFactoryImpl);

} // namespace 
} // namespace primitive
} // namespace oneflow