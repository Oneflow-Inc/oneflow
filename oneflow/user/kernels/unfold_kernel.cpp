// #include "oneflow/core/framework/framework.h"
// #include "oneflow/user/ops/nn_util.h"
// #include "oneflow/core/operator/operator_util.h"
// #include "oneflow/user/kernels/unfold_kernel_util.h"

// namespace oneflow {

// namespace user_op {

// namespace {

// template<typename INDEX_T, int NDIM, int SDIM>
// class UnfoldOpKernelState : public OpKernelState {
//  public:
//   using ParamType = UnfoldParams<INDEX_T, NDIM, SDIM>;
//   UnfoldOpKernelState(const ShapeView& input_shape, const std::vector<int32_t>& kernel_size,
//                       const std::vector<int32_t>& padding, const std::vector<int32_t>& stride,
//                       const std::vector<int32_t>& dilation)
//       : params_(input_shape.At(0), input_shape.At(ParamType::kInputChannelDim),
//                 input_shape.ptr() + SDIM, kernel_size.data(), padding.data(),
//                 stride.data(), dilation.data()) {}
//   const ParamType& params() const { return params_; }

//  private:
//   ParamType params_;
// };

// template<typename INDEX_T, int NDIM, int SDIM>
// std::shared_ptr<OpKernelState> CreateUnfoldOpKernelState(const ShapeView& input_shape,
//                                                          const std::vector<int32_t>& kernel_size,
//                                                          const std::vector<int32_t>& padding,
//                                                          const std::vector<int32_t>& stride,
//                                                          const std::vector<int32_t>& dilation) {
//   return std::make_shared<UnfoldOpKernelState<INDEX_T, NDIM, SDIM>>(
//       input_shape, kernel_size, padding, stride, dilation);
// }

// template<typename INDEX_T, int NDIM, int SDIM>
// const void* GetUnfoldParams(OpKernelState* state) {
//   auto* unfold_state = dynamic_cast<UnfoldOpKernelState<INDEX_T, NDIM, SDIM>*>(state);
//   CHECK_NOTNULL(unfold_state);
//   return static_cast<const void*>(&unfold_state->params());
// }

// // #define SWITCH_ENTRY(func_name, itype, ndim, sdim) func_name<itype, ndim, sdim>
// // #define DEFINE_UNFOLD_SWITCH_FUNC(ret_type, func_name)                                 \
// //   DEFINE_STATIC_SWITCH_FUNC(                                                           \
// //       ret_type, func_name, SWITCH_ENTRY, MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ), \
// //       MAKE_NDIM_CTRV_SEQ(SPATIAL_NDIM_SEQ), MAKE_NDIM_CTRV_SEQ(SPATIAL_DIM_SEQ));
// // DEFINE_UNFOLD_SWITCH_FUNC(std::shared_ptr<OpKernelState>, CreateUnfoldOpKernelState);
// // DEFINE_UNFOLD_SWITCH_FUNC(const void*, GetUnfoldParams);
// // #undef DEFINE_UNFOLD_SWITCH_FUNC
// // #undef SWITCH_ENTRY

// template<DeviceType device_type, typename T>
// class UnfoldKernel final : public OpKernel {
//  public:
//   UnfoldKernel() = default;
//   ~UnfoldKernel() = default;

//   std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const override {
//     const TensorDesc* input_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
//     if (input_desc->is_dynamic()) { return std::shared_ptr<OpKernelState>(nullptr); }
//     const std::string data_format = ctx->Attr<std::string>("data_format");
//     const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
//     const std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding");
//     const std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("strides");
//     const std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
//     const int spatial_ndim = input_desc->shape().NumAxes() - 2;
//     const int spatial_dim = GetSpatialDim(data_format);
//     const DataType index_dtype = input_desc->shape().elem_cnt() < std::numeric_limits<int32_t>::max()
//                                ? DataType::kInt32
//                                : DataType::kInt64;
    
//     OpKernelState* state = CreateUnfoldOpKernelState<index_dtype, spatial_ndim, spatial_dim>(ShapeView(input_desc->shape()), kernel_size, padding, stride, dilation);
//     return GetUnfoldParams<index_dtype, spatial_ndim, spatial_dim>(state);
//   }

//  private:
//   int GetSpatialDim(const std::string& data_format) const {
//     int spatial_dim = 0;
//     if (data_format == "channels_first") {
//       spatial_dim = 2;
//     } else if (data_format == "channels_last") {
//       spatial_dim = 1;
//     } else {
//       UNIMPLEMENTED();
//     }
//     return spatial_dim;
//   }

// // #define SWITCH_ENTRY(func_name, itype, ndim, sdim) \
// //   UnfoldKernelUtil<device_type, T, itype, ndim, sdim>::func_name
// //   DEFINE_STATIC_SWITCH_FUNC(void, Forward, SWITCH_ENTRY,
// //                             MAKE_DATA_TYPE_CTRV_SEQ(INDEX_DATA_TYPE_SEQ),
// //                             MAKE_NDIM_CTRV_SEQ(SPATIAL_NDIM_SEQ),
// //                             MAKE_NDIM_CTRV_SEQ(SPATIAL_DIM_SEQ));
// // #undef SWITCH_ENTRY

//   void Compute(KernelComputeContext* ctx, OpKernelState* state) const override {
//     const Tensor* input = ctx->Tensor4ArgNameAndIndex("x", 0);
//     Tensor* output = ctx->Tensor4ArgNameAndIndex("y", 0);
//     const int spatial_ndim = input->shape().NumAxes() - 2;
//     const int spatial_dim = GetSpatialDim(ctx->Attr<std::string>("data_format")); 
//     DataType index_dtype = input->shape().elem_cnt() < std::numeric_limits<int32_t>::max()
//                                ? DataType::kInt32
//                                : DataType::kInt64;
    
//     // auto switch_case = SwitchCase(index_dtype, spatial_ndim, spatial_dim);

//     std::shared_ptr<OpKernelState> state_ptr(nullptr);
//     if (state == nullptr) {
//       const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
//       const std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding");
//       const std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("strides");
//       const std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
//       state_ptr = CreateUnfoldOpKernelState<index_dtype, spatial_ndim, spatial_dim>(input->shape(), kernel_size,
//                                             padding, stride, dilation);
//       state = state_ptr.get();
//     }
//     const void* params = GetUnfoldParams<index_dtype, spatial_ndim, spatial_dim>(state);
//     UnfoldKernelUtil<index_dtype, spatial_ndim, spatial_dim>::Forward(ctx->device_ctx(), params, input->dptr<T>(), output->mut_dptr<T>());
//   }

//   bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
// };


// }  // namespace

// #define REGISTER_UNFOLD_KERNEL(device, dtype)                                                \
//   REGISTER_USER_KERNEL("unfold").SetCreateFn<UnfoldKernel<device, dtype>>().SetIsMatchedHob( \
//       (user_op::HobDeviceTag() == device)                                                    \
//       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));                        

// REGISTER_UNFOLD_KERNEL(DeviceType::kCPU, float)
// REGISTER_UNFOLD_KERNEL(DeviceType::kCPU, double)

// #ifdef WITH_CUDA
// REGISTER_UNFOLD_KERNEL(DeviceType::kGPU, float)
// REGISTER_UNFOLD_KERNEL(DeviceType::kGPU, double)
// #endif  // WITH_CUDA

// }  // namespace user_op

// }  // namespace oneflow




#include "oneflow/core/framework/framework.h"
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/user/kernels/unfold_kernel_util.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename INDEX_T, int NDIM, int SDIM>
class UnfoldOpKernelState : public OpKernelState {
 public:
  using ParamType = UnfoldParams<INDEX_T, NDIM, SDIM>;
  UnfoldOpKernelState(const ShapeView& input_shape, const std::vector<int32_t>& kernel_size,
                      const std::vector<int32_t>& padding, const std::vector<int32_t>& stride,
                      const std::vector<int32_t>& dilation)
      : params_(input_shape.At(0), input_shape.At(ParamType::kInputChannelDim),
                input_shape.ptr() + SDIM, kernel_size.data(), padding.data(),
                stride.data(), dilation.data()) {}
  const ParamType& params() const { return params_; }

 private:
  ParamType params_;
};

template<typename INDEX_T, int NDIM, int SDIM>
std::shared_ptr<UnfoldOpKernelState<INDEX_T, NDIM, SDIM>> CreateUnfoldOpKernelState(const ShapeView& input_shape,
                                                         const std::vector<int32_t>& kernel_size,
                                                         const std::vector<int32_t>& padding,
                                                         const std::vector<int32_t>& stride,
                                                         const std::vector<int32_t>& dilation) {
  // return std::make_shared<UnfoldOpKernelState<INDEX_T, NDIM, SDIM>>(
  //     input_shape, kernel_size, padding, stride, dilation);
  std::shared_ptr<UnfoldOpKernelState<INDEX_T, NDIM, SDIM>> state(new UnfoldOpKernelState<INDEX_T, NDIM, SDIM>(input_shape, kernel_size, padding, stride, dilation));
  return state;
}

template<DeviceType device_type, typename T, typename INDEX_T, int NDIM, int SDIM>
class UnfoldKernel final : public OpKernel {
 public:
  UnfoldKernel() = default;
  ~UnfoldKernel() = default;

  // std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const override {
  //   const TensorDesc* input_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  //   if (input_desc->is_dynamic()) { return std::shared_ptr<OpKernelState>(nullptr); }
  //   const std::string data_format = ctx->Attr<std::string>("data_format");
  //   const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  //   const std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding");
  //   const std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("strides");
  //   const std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
  //   const int spatial_ndim = input_desc->shape().NumAxes() - 2;
  //   const int spatial_dim = GetSpatialDim(data_format);
  //   const DataType index_dtype = input_desc->shape().elem_cnt() < std::numeric_limits<int32_t>::max()
  //                              ? DataType::kInt32
  //                              : DataType::kInt64;
    
  //   OpKernelState* state = CreateUnfoldOpKernelState<INDEX_T, NDIM, SDIM>(ShapeView(input_desc->shape()), kernel_size, padding, stride, dilation);
  //   return GetUnfoldParams<INDEX_T, NDIM, SDIM>(state);
  // }

 private:
  int GetSpatialDim(const std::string& data_format) const {
    int spatial_dim = 0;
    if (data_format == "channels_first") {
      spatial_dim = 2;
    } else if (data_format == "channels_last") {
      spatial_dim = 1;
    } else {
      UNIMPLEMENTED();
    }
    return spatial_dim;
  }

  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* input = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* output = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int spatial_ndim = input->shape().NumAxes() - 2;
    const int spatial_dim = GetSpatialDim(ctx->Attr<std::string>("data_format")); 
    // DataType index_dtype = input->shape().elem_cnt() < std::numeric_limits<int32_t>::max()
    //                            ? DataType::kInt32
    //                            : DataType::kInt64;
    
    // auto switch_case = SwitchCase(index_dtype, spatial_ndim, spatial_dim);

    // std::shared_ptr<UnfoldOpKernelState<INDEX_T, NDIM, SDIM>> state_ptr(nullptr);
    const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding");
    const std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("strides");
    const std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    
    
    const auto& state_ptr = CreateUnfoldOpKernelState<INDEX_T, NDIM, SDIM>(input->shape(), kernel_size,
                                          padding, stride, dilation);
   
    const UnfoldParams<INDEX_T, NDIM, SDIM> params = state_ptr->params();
    printf("output elemcnt is: %d", params.out_elem_cnt); 
    UnfoldKernelUtil<device_type, T, INDEX_T, NDIM, SDIM>::Forward(ctx->device_ctx(), &params, input->dptr<T>(), output->mut_dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};


}  // namespace

#define REGISTER_UNFOLD_KERNEL(device, dtype)                                                \
  REGISTER_USER_KERNEL("unfold").SetCreateFn<UnfoldKernel<device, dtype, int32_t, 2, 1>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                                    \
      & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));                        

REGISTER_UNFOLD_KERNEL(DeviceType::kCPU, float)
REGISTER_UNFOLD_KERNEL(DeviceType::kCPU, double)

#ifdef WITH_CUDA
REGISTER_UNFOLD_KERNEL(DeviceType::kGPU, float)
REGISTER_UNFOLD_KERNEL(DeviceType::kGPU, double)
#endif  // WITH_CUDA

}  // namespace user_op

}  // namespace oneflow