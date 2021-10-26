#include "oneflow/core/primitive/include/unary_op.h"
#include "oneflow/core/primitive/cpu/type_seq.h"

namespace oneflow{

namespace primitive{

template<DeviceType device, UnaryOpList unary_enum, typename Out, typename In>
struct UnaryFunctor; 

template<typename Out, typename In>
struct UnaryFunctor<DeviceType::kCPU, UnaryOpList::kRelu, Out, In>{
  Out operator()(In src) const {
    return src ? src > static_cast<In>(0) : static_cast<Out>(0); 
  }
}; 

namespace{

template<UnaryOpList unary_enum, typename Out, typename In>
class ElementwiseUnaryOpImpl: public ElementwiseUnaryOp{
public: 
    OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryOpImpl); 
    ElementwiseUnaryOpImpl() = default;
    ~ElementwiseUnaryOpImpl() override = default; 

    void Launch(StreamContext* stream_ctx, size_t count, void* dst_ptr, const void* src_ptr) override{
        Out* dst = reinterpret_cast<Out*>(dst_ptr); 
        const In* src = reinterpret_cast<const In*>(src_ptr); 
        for (size_t i = 0; i < count; ++i) {dst[i] = UnaryFunctor<DeviceType::kCPU, unary_enum, Out, In>()(src[i]); }
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
            MAKE_NEW_UNARYOP_ENTRY, CPU_PRIMITIVE_UNARY_OP_SEQ, CPU_PRIMITIVE_NATIVE_TYPE_SEQ, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)};

#undef MAKE_NEW_UNARYOP_ENTRY

    const auto it = new_elementwise_unary_op_handle.find(std::make_tuple(op_enum, out, in));
    if (it != new_elementwise_unary_op_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, ElementwiseUnaryOpFactory, ElementwiseUnaryOpFactoryImpl);

} // namespace 
} // namespace primitive
} // namespace oneflow