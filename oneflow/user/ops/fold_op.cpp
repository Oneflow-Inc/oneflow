// #include "oneflow/core/framework/framework.h"

// namespace oneflow {

// REGISTER_USER_OP("fold")
//     .Input("x")
//     .Output("y")
//     .Attr<std::vector<int32_t>>("output_size")
//     .Attr<std::vector<int32_t>>("kernel_size")
//     .Attr<std::vector<int32_t>>("strides")
//     .Attr<std::vector<int32_t>>("padding")
//     .Attr<std::vector<int32_t>>("dilation_rate")
//     .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
//       const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
//       user_op::TensorDesc* y = ctx->OutputTensorDesc("y", 0);
      
//       const std::vector<int32_t> output_size = ctx->Attr<std::vector<int32_t>>("output_size");
//       const std::vector<int32_t> padding = ctx->Attr<std::vector<int32_t>>("padding");
//       const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
//       const std::vector<int32_t>& strides = ctx->Attr<std::vector<int32_t>>("strides");
//       const std::vector<int32_t>& dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
      
//       const Shape& x_shape = x.shape();
//       const int32_t batch = x_shape.At(0); 
//       const int32_t channels_k_k = x_shape.At(1); // channels * kernel_size[0] * kernel_size[1]
//       const int32_t window_num = x_shape.At(2); 

//       const int32_t ndim = x_shape.NumAxes(); 
//       CHECK_EQ_OR_RETURN(ndim, 3);  

//       // todo: !!!!!!!!!!!!!============= !!!!!!!
      
//       for (int32_t kernel_dim : kernel_size) { CHECK_GT_OR_RETURN(kernel_dim, 0); }
//       CHECK_EQ_OR_RETURN(strides.size(), 2);
//       for (int32_t stride_dim : strides) { CHECK_GT_OR_RETURN(stride_dim, 0); }
//       CHECK_EQ_OR_RETURN(dilation_rate.size(), 2);
//       for (int32_t dilation_dim : dilation_rate) { CHECK_GT_OR_RETURN(dilation_dim, 0); }
      
//       const int32_t y_height = (x_height + 2*padding[0] - dilation_rate[0] * (kernel_size[0] - 1) - 1)/ strides[0] + 1;
//       const int32_t y_width = (x_width + 2*padding[1] - dilation_rate[1] * (kernel_size[1] - 1) - 1)/ strides[1] + 1;
     
//       DimVector y_shape(3);
//       y_shape.at(0) = batch; 
//       y_shape.at(1) = in_channels * kernel_size[0] * kernel_size[1]; 
//       y_shape.at(2) = y_height * y_width; 
//       *y->mut_shape() = Shape(y_shape);

//       return Maybe<void>::Ok();
//     })
//     .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
//       *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
//       return Maybe<void>::Ok();
//     })
//     .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
//       ctx->NewBuilder()
//           .Split(user_op::OpArg("x", 0), 0)
//           .Split(user_op::OpArg("y", 0), 0)
//           .Build();
//       return Maybe<void>::Ok();
//     });

// }  // namespace oneflow