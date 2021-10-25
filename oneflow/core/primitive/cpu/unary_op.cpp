#include "oneflow/core/primitive/include/unary_op.h"
#include <memory>
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/primitive/cpu/type_seq.h"
#include "oneflow/core/stream/stream_context.h"
#include "oneflow/user/kernels/elementwise_xpu_kernel.h"

namespace oneflow{

namespace primitive{

namespace{

// template<template<typename Out, typename In> class FunctorT, typename Out, typename In>
// class UnaryOpImpl: public UnaryOp{
// public: 
//     OF_DISALLOW_COPY_AND_MOVE(UnaryOpImpl); 
//     UnaryOpImpl() = default;
//     ~UnaryOpImpl() override = default; 

//     void Launch(StreamContext* stream_ctx, size_t count, void* dst_ptr, const void* src_ptr) override{
//         Out* dst = reinterpret_cast<Out*>(dst_ptr); 
//         const In* src = reinterpret_cast<const In*>(src_ptr); 
//         FunctorT<Out, In>()(dst, src, count);
//     }
// }; 

// template<template<typename Out, typename In> class FunctorT, typename Out, typename In>
// std::unique_ptr<UnaryOp> NewUnaryOp() {
//   return std::unique_ptr<UnaryOp>(new UnaryOpImpl<FunctorT, Out, In>());
// }

// class UnaryOpFactoryImpl : public UnaryOpFactory {
//  public:
//   OF_DISALLOW_COPY_AND_MOVE(UnaryOpFactoryImpl);
//   UnaryOpFactoryImpl() = default;
//   ~UnaryOpFactoryImpl() override = default;


//   std::unique_ptr<UnaryOp> New(UnaryOpList op_enum, DataType out, DataType in) override {
// #define MAKE_NEW_UNARYOP_ENTRY(op_pair, out_pair, in_pair)                              \
//   {std::make_tuple(OF_PP_PAIR_SECOND(op_pair), OF_PP_PAIR_SECOND(out_pair), OF_PP_PAIR_SECOND(in_pair)), \
//    NewUnaryOp<OF_PP_PAIR_FIRST(op_pair), OF_PP_PAIR_FIRST(out_pair), OF_PP_PAIR_FIRST(in_pair)>},

//     static const std::map<std::tuple<UnaryOpList, DataType, DataType>, std::function<std::unique_ptr<UnaryOp>()>>
//         new_unary_op_handle{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
//             MAKE_NEW_UNARYOP_ENTRY, CPU_PRIMITIVE_UNARY_OP_SEQ, CPU_PRIMITIVE_NATIVE_TYPE_SEQ, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)};

// #undef MAKE_NEW_UNARYOP_ENTRY

//     const auto it = new_unary_op_handle.find(std::make_tuple(op_enum, out, in));
//     if (it != new_unary_op_handle.end()) {
//       return it->second();
//     } else {
//       return nullptr;
//     }
//   }
// };

// REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, UnaryOpFactory, UnaryOpFactoryImpl);

template<DeviceType device, UnaryOpList unary_enum, typename Out, typename In>
struct UnaryFunctor; 

template<typename Out, typename In>
struct UnaryFunctor<DeviceType::kCPU, UnaryOpList::kIdentity, Out, In>{
  Out operator()(In src) const {
    return src; 
  }
}; 


template<UnaryOpList unary_enum, typename Out, typename In>
class UnaryOpImpl: public UnaryOp{
public: 
    OF_DISALLOW_COPY_AND_MOVE(UnaryOpImpl); 
    UnaryOpImpl() = default;
    ~UnaryOpImpl() override = default; 

    void Launch(StreamContext* stream_ctx, size_t count, void* dst_ptr, const void* src_ptr) override{
        Out* dst = reinterpret_cast<Out*>(dst_ptr); 
        const In* src = reinterpret_cast<const In*>(src_ptr); 
        for (size_t i = 0; i < count; ++i) {dst[i] = UnaryFunctor<DeviceType::kCPU, unary_enum, Out, In>()(src[i]); }
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
            MAKE_NEW_UNARYOP_ENTRY, CPU_PRIMITIVE_UNARY_OP_SEQ, CPU_PRIMITIVE_NATIVE_TYPE_SEQ, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)};

#undef MAKE_NEW_UNARYOP_ENTRY

    const auto it = new_unary_op_handle.find(std::make_tuple(op_enum, out, in));
    if (it != new_unary_op_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, UnaryOpFactory, UnaryOpFactoryImpl);

}
}
}