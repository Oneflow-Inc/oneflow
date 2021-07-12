#ifdef WITH_CUDA
#include "oneflow/user/kernels/dim_scatter_scalar.h"

namespace oneflow{

namespace user_op{

namespace{

template<typename IN_T, typename IDX_T>                                                          
__global__ void DoCUDADimScatterScalarUpdate(const DimOpIndexNdHelper<IDX_T> idx_nd_helper, 
                                       const DimOpIndexNdHelper<IDX_T> output_nd_helper,        
                                       const int ndim, const int64_t elem_cnt, const int32_t dim, const int64_t upper_bound, 
                                       const IDX_T* index, const IN_T src_scalar, IN_T* output) {   
    ScatterScalarUpdateFunctor<IN_T, IDX_T>(idx_nd_helper, output_nd_helper, ndim, elem_cnt, dim, upper_bound, index, src_scalar, output);
    }                   
} // namespace 

template<DeviceType device_type, typename IN_T, typename IDX_T>                                 
class GpuDimScatterScalarUpdateKernel final : public OpKernel { 
  public:                                                                                        
  GpuDimScatterScalarUpdateKernel() = default;                                                        
  ~GpuDimScatterScalarUpdateKernel() = default;                                              
                                                                                              
  private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("input", 0);
    const Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("output", 0);
    const int32_t dim = ctx->Attr<int32_t>("dim");

    const IDX_T* index = index_tensor->dptr<IDX_T>();
    IN_T* output = out_tensor->mut_dptr<IN_T>();
    size_t out_bytes_size =
        out_tensor->shape().elem_cnt() * GetSizeOfDataType(out_tensor->data_type());

    Tensor* like_tensor = ctx->Tensor4ArgNameAndIndex("like", 0);
    const IN_T src_scalar = static_cast<IN_T>(ctx->Attr<float>("src_scalar"));

    if (input_tensor) {
      Memcpy<device_type>(ctx->device_ctx(), output, input_tensor->dptr<IN_T>(), out_bytes_size);
    } else if (like_tensor) {
      Memset<device_type>(ctx->device_ctx(), output, 0, out_bytes_size);
    } else {
      std::cout<<"Unimplemented Error"<<std::endl;
      throw Error::Unimplemented();
    }

    const int ndim = out_tensor->shape().NumAxes();
    fixed_vector<IDX_T, kDimGatherMaxDimCount> shape_vec(ndim);
    auto shape2dims = [&shape_vec, &ndim](const ShapeView& tensor_shape) -> void {
      std::transform(tensor_shape.ptr(), tensor_shape.ptr() + ndim, shape_vec.begin(),
                      [](int32_t dim) -> IDX_T { return static_cast<IDX_T>(dim); });
    };
    shape2dims(index_tensor->shape());
    DimOpIndexNdHelper<IDX_T> idx_nd_helper(shape_vec.data(), ndim);
    shape2dims(out_tensor->shape());
    DimOpIndexNdHelper<IDX_T> output_nd_helper(shape_vec.data(), ndim);

    int64_t upper_bound = input_tensor->shape().At(dim);
    int64_t elem_cnt = index_tensor->shape().elem_cnt(); 

    RUN_CUDA_KERNEL((DoCUDADimScatterScalarUpdate<IN_T, IDX_T>), ctx->device_ctx(), BlocksNum4ThreadsNum(elem_cnt),
                    idx_nd_helper, output_nd_helper, ndim, elem_cnt, dim, upper_bound, index,
                    src_scalar, output);

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }                      
};

#define REGISTER_GPU_SCATTERSCALAR_KERNEL(device, dtype, itype)                          \
  REGISTER_USER_KERNEL("dim_scatter_scalar_update")                                      \
      .SetCreateFn<GpuDimScatterScalarUpdateKernel<device, dtype, itype>>()              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("index", 0) == GetDataType<itype>::value));   

REGISTER_GPU_SCATTERSCALAR_KERNEL(DeviceType::kGPU, float, int32_t);
REGISTER_GPU_SCATTERSCALAR_KERNEL(DeviceType::kGPU, float, int64_t);
REGISTER_GPU_SCATTERSCALAR_KERNEL(DeviceType::kGPU, float16, int32_t);
REGISTER_GPU_SCATTERSCALAR_KERNEL(DeviceType::kGPU, float16, int64_t);
REGISTER_GPU_SCATTERSCALAR_KERNEL(DeviceType::kGPU, double, int32_t);
REGISTER_GPU_SCATTERSCALAR_KERNEL(DeviceType::kGPU, double, int64_t);


} // namespace user_op
} // namespace oneflow 
#endif
