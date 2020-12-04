#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/user/kernels/ravel_multi_index_util.h"

namespace oneflow {
 
namespace user_op {

template<DeviceType device_type, typename T>
class RavelMultiIndexKernel final : public OpKernel {
 public:
  RavelMultiIndexKernel() = default;
  ~RavelMultiIndexKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* dims_tensor = ctx->Tensor4ArgNameAndIndex("dims", 0);
    int ndim = dims_tensor->shape().elem_cnt();
    size_t in_num = ctx->inputs().size() - 1; // ([3, 6, 2], [4, 5, 1]) -> in_num 还有个额外的输入 dim = 3, 所以要减1
    
    std::vector<const T*> in_dptrs(in_num);
    for (int32_t i = 0; i < in_num; ++i) {
      std::cout<<"Current Loop idx is: "<<i<<std::endl;
      in_dptrs.at(i) = ctx->Tensor4ArgNameAndIndex("multi_index", i)->dptr<T>();
    }

    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int n = out->shape().elem_cnt();
    T* output = out->mut_dptr<T>();

    const T* dims = dims_tensor->dptr<T>();
    std::cout<<"Helper Ndim is: "<<ndim<<std::endl;
    RavelMultiIndexHelper<T> helper(dims, ndim);

    // for (int32_t elem_idx = 0; elem_idx < n; ++elem_idx){
    //     std::vector<T> index_vec(in_num);
    //     std::cout<<"Index vector size is: "<<in_num<<std::endl;
    //     for (int32_t idx = 0; idx < in_num; ++idx){
    //       std::cout<<"In dptrs element is: "<<in_dptrs.at(idx)[elem_idx]<<std::endl;
    //       index_vec[idx] = (in_dptrs.at(idx)[elem_idx]);
    //     }
    //     std::cout<<"Index vector size is: "<<index_vec[0]<<std::endl;
    //     std::cout<<"Index vector size is: "<<index_vec[1]<<std::endl;
    //     std::cout<<"n is: "<<n<<std::endl;
    //     T offset = helper.NdIndexToOffset(index_vec.data(), in_num);
    //     std::cout<<"offset is: "<<offset<<std::endl;
    //     output[elem_idx] = offset;
    //   }

    RavelMultiIndexFunctor<device_type, T>()(ctx->device_ctx(), n, in_num, helper, in_dptrs, output);

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RAVEL_MULTI_INDEX_KERNEL(device, dtype)    \
  REGISTER_USER_KERNEL("ravel_multi_index")                 \
      .SetCreateFn<RavelMultiIndexKernel<device, dtype>>()  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) & \
                        (user_op::HobDataType("dims", 0) == GetDataType<dtype>::value));

#define REGISTER_RAVEL_MULTI_INDEX_KERNELS_WITH_DEVICE(device) \
  REGISTER_RAVEL_MULTI_INDEX_KERNEL(device, int32_t)           \
  REGISTER_RAVEL_MULTI_INDEX_KERNEL(device, int64_t)           \

// Register CPU version
REGISTER_RAVEL_MULTI_INDEX_KERNELS_WITH_DEVICE(DeviceType::kCPU);

// Register GPU version

#ifdef WITH_CUDA
REGISTER_RAVEL_MULTI_INDEX_KERNELS_WITH_DEVICE(DeviceType::kGPU);
#endif

}  // namespace user_op
}  // namespace oneflow
