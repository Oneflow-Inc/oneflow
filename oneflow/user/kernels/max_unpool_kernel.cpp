/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/user/kernels/max_unpool_kernel_util.h"

namespace oneflow {

// struct UnpoolOpKernelCache final : public user_op::OpKernelCache {
//   MaxUnpoolParams3D params_3d;
//   explicit UnpoolOpKernelCache(const MaxUnpoolParams3D& params_3d) : params_3d(params_3d) {}
//   const MaxUnpoolParams3D& GetParams3D() const { return params_3d; }
// };

// std::shared_ptr<UnpoolOpKernelCache> CreateUnpoolOpKernelCache(user_op::KernelCacheContext* ctx,
//                                                                const int32_t& dim) {
//   const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
//   const std::string& data_format = ctx->Attr<std::string>("data_format");
//   const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
//   const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
//   const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
//   MaxUnpoolParams3D params_3d =
//       MaxUnpoolParams3D(dim, x_shape, data_format, padding, kernel_size, stride);
//   std::shared_ptr<UnpoolOpKernelCache> cache(new UnpoolOpKernelCache(params_3d));
//   return cache;
// }

template<typename T, typename IDX>
struct UnpoolKernelUtil<DeviceType::kCPU, T, IDX> {
  static void MaxUnpoolNdForward(ep::Stream* stream,
                                 const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                 const IDX elem_num, const T* src, T* dest,
                                 const int64_t* indice_ptr, const int64_t y_hwd_size) {
    XPU_1D_KERNEL_LOOP(num, elem_num) {
      IDX bc_idx, hwd_idx;
      index_helper.OffsetToNdIndex(num, bc_idx, hwd_idx);
      IDX dest_idx = bc_idx * y_hwd_size + indice_ptr[num];
      dest[dest_idx] = src[num];
    }
  }

  static void MaxUnpoolNdBackward(ep::Stream* stream,
                                  const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                  const IDX elem_num, const T* src, T* dest,
                                  const int64_t* indice_ptr, const int64_t dx_hwd_size) {
    XPU_1D_KERNEL_LOOP(num, elem_num) {
      IDX bc_idx, hwd_idx;
      index_helper.OffsetToNdIndex(num, bc_idx, hwd_idx);
      IDX dest_idx = bc_idx * dx_hwd_size + indice_ptr[num];
      dest[dest_idx] = src[num];
    }
  }
};

template<DeviceType device_type, typename T>
class MaxUnpool1dKernel final : public user_op::OpKernel {
 public:
  MaxUnpool1dKernel() = default;
  ~MaxUnpool1dKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    const int64_t elem_num = x->shape_view().elem_cnt();
    const T* src = x->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = y->mut_dptr<T>();

    DimVector x_vector(2);
    x_vector.at(0) = x->shape_view().At(0) * x->shape_view().At(1);
    x_vector.at(1) = x->shape_view().At(2);
    const int64_t y_hwd_size = y->shape_view().At(2);

    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
    CHECK(memset_primitive);
    memset_primitive->Launch(ctx->stream(), dest, 0, y->shape_view().elem_cnt());

    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 2> index_helper(x_vector.data());
      UnpoolKernelUtil<device_type, T, int32_t>::MaxUnpoolNdForward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, y_hwd_size);
    } else {
      NdIndexOffsetHelper<int64_t, 2> index_helper(x_vector.data());
      UnpoolKernelUtil<device_type, T, int64_t>::MaxUnpoolNdForward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, y_hwd_size);
    }
  }
};

template<DeviceType device_type, typename T>
class MaxUnpool1dGradKernel final : public user_op::OpKernel {
 public:
  MaxUnpool1dGradKernel() = default;
  ~MaxUnpool1dGradKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const int64_t elem_num = dy->shape_view().elem_cnt();
    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();
    DimVector dy_vector(2);
    dy_vector.at(0) = dy->shape_view().At(0) * dy->shape_view().At(1);
    dy_vector.at(1) = dy->shape_view().At(2);

    const int64_t dx_hwd_size = dx->shape_view().At(2);
    std::unique_ptr<ep::primitive::Memset> memset_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
    CHECK(memset_primitive);
    memset_primitive->Launch(ctx->stream(), dest, 0, dx->shape_view().elem_cnt());

    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 2> index_helper(dy_vector.data());
      UnpoolKernelUtil<device_type, T, int32_t>::MaxUnpoolNdBackward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, dx_hwd_size);
    } else {
      NdIndexOffsetHelper<int64_t, 2> index_helper(dy_vector.data());
      UnpoolKernelUtil<device_type, T, int64_t>::MaxUnpoolNdBackward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, dx_hwd_size);
    }
  };
};

// template<DeviceType device_type, typename T>
// class MaxPool2dKernel final : public user_op::OpKernel {
//  public:
//   MaxPool2dKernel() = default;
//   ~MaxPool2dKernel() = default;

//   bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
//   std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
//       user_op::KernelCacheContext* ctx) const override {
//     return CreatePoolOpKernelCache(ctx, 2);
//   }

//  private:
//   void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
//                const user_op::OpKernelCache* cache) const override {
//     const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
//     user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
//     user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);

//     const auto* pool_cache = dynamic_cast<const PoolOpKernelCache*>(cache);
//     const MaxPoolParams3D& params_3d = pool_cache->GetParams3D();

//     const int64_t elem_num = y->shape_view().elem_cnt();

//     const T* src = x->dptr<T>();
//     T* dest = y->mut_dptr<T>();
//     int64_t* indice_ptr = indice->mut_dptr<int64_t>();

//     const std::string& data_format = ctx->Attr<std::string>("data_format");
//     if (data_format == "channels_first") {
//       DimVector y_vector(3);
//       y_vector.at(0) = y->shape_view().At(0) * y->shape_view().At(1);
//       y_vector.at(1) = y->shape_view().At(2);
//       y_vector.at(2) = y->shape_view().At(3);
//       if (elem_num < GetMaxVal<int32_t>()) {
//         NdIndexOffsetHelper<int32_t, 3> index_helper(y_vector.data());
//         PoolKernelUtil<device_type, T, int32_t>::Maxpool2dForwardCFirst(
//             ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
//       } else {
//         NdIndexOffsetHelper<int64_t, 3> index_helper(y_vector.data());
//         PoolKernelUtil<device_type, T, int64_t>::Maxpool2dForwardCFirst(
//             ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
//       }
//     } else if (data_format == "channels_last") {
//       DimVector y_vector;
//       y->shape_view().ToDimVector(&y_vector);
//       if (elem_num < GetMaxVal<int32_t>()) {
//         NdIndexOffsetHelper<int32_t, 4> index_helper(y_vector.data());
//         PoolKernelUtil<device_type, T, int32_t>::Maxpool2dForwardCLast(
//             ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
//       } else {
//         NdIndexOffsetHelper<int64_t, 4> index_helper(y_vector.data());
//         PoolKernelUtil<device_type, T, int64_t>::Maxpool2dForwardCLast(
//             ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
//       }
//     } else {
//       UNIMPLEMENTED() << "Unsupported data_format";
//     }
//   };
// };

// template<DeviceType device_type, typename T>
// class MaxPool2dGradKernel final : public user_op::OpKernel {
//  public:
//   MaxPool2dGradKernel() = default;
//   ~MaxPool2dGradKernel() = default;

//   bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
//   std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
//       user_op::KernelCacheContext* ctx) const override {
//     return CreatePoolOpKernelCache(ctx, 2);
//   }

//  private:
//   void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
//                const user_op::OpKernelCache* cache) const override {
//     const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
//     const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
//     user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

//     const auto* pool_cache = dynamic_cast<const PoolOpKernelCache*>(cache);
//     const MaxPoolParams3D& params_3d = pool_cache->GetParams3D();

//     const int64_t elem_num = dy->shape_view().elem_cnt();
//     const T* src = dy->dptr<T>();
//     const int64_t* indice_ptr = indice->dptr<int64_t>();
//     T* dest = dx->mut_dptr<T>();

//     size_t out_bytes_size = dx->shape_view().elem_cnt() * GetSizeOfDataType(dx->data_type());
//     Memset<device_type>(ctx->stream(), dest, 0, out_bytes_size);

//     const std::string& data_format = ctx->Attr<std::string>("data_format");

//     if (data_format == "channels_first") {
//       DimVector dy_vector(3);
//       dy_vector.at(0) = dy->shape_view().At(0) * dy->shape_view().At(1);
//       dy_vector.at(1) = dy->shape_view().At(2);
//       dy_vector.at(2) = dy->shape_view().At(3);
//       if (elem_num < GetMaxVal<int32_t>()) {
//         NdIndexOffsetHelper<int32_t, 3> index_helper(dy_vector.data());
//         PoolKernelUtil<device_type, T, int32_t>::Maxpool2dBackwardCFirst(
//             ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
//       } else {
//         NdIndexOffsetHelper<int64_t, 3> index_helper(dy_vector.data());
//         PoolKernelUtil<device_type, T, int64_t>::Maxpool2dBackwardCFirst(
//             ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
//       }
//     } else if (data_format == "channels_last") {
//       DimVector dy_vector;
//       dy->shape_view().ToDimVector(&dy_vector);
//       if (elem_num < GetMaxVal<int32_t>()) {
//         NdIndexOffsetHelper<int32_t, 4> index_helper(dy_vector.data());
//         PoolKernelUtil<device_type, T, int32_t>::Maxpool2dBackwardCLast(
//             ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
//       } else {
//         NdIndexOffsetHelper<int64_t, 4> index_helper(dy_vector.data());
//         PoolKernelUtil<device_type, T, int64_t>::Maxpool2dBackwardCLast(
//             ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
//       }
//     } else {
//       UNIMPLEMENTED() << "Unsupported data_format";
//     }
//   };
// };

// template<DeviceType device_type, typename T>
// class MaxPool3dKernel final : public user_op::OpKernel {
//  public:
//   MaxPool3dKernel() = default;
//   ~MaxPool3dKernel() = default;

//   bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
//   std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
//       user_op::KernelCacheContext* ctx) const override {
//     return CreatePoolOpKernelCache(ctx, 3);
//   }

//  private:
//   void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
//                const user_op::OpKernelCache* cache) const override {
//     const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
//     user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
//     user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);

//     const auto* pool_cache = dynamic_cast<const PoolOpKernelCache*>(cache);
//     const MaxPoolParams3D& params_3d = pool_cache->GetParams3D();

//     const int64_t elem_num = y->shape_view().elem_cnt();
//     const T* src = x->dptr<T>();
//     T* dest = y->mut_dptr<T>();
//     int64_t* indice_ptr = indice->mut_dptr<int64_t>();

//     DimVector y_vector(4);
//     y_vector.at(0) = y->shape_view().At(0) * y->shape_view().At(1);
//     y_vector.at(1) = y->shape_view().At(2);
//     y_vector.at(2) = y->shape_view().At(3);
//     y_vector.at(3) = y->shape_view().At(4);

//     if (elem_num < GetMaxVal<int32_t>()) {
//       NdIndexOffsetHelper<int32_t, 4> index_helper(y_vector.data());
//       PoolKernelUtil<device_type, T, int32_t>::Maxpool3dForward(
//           ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
//     } else {
//       NdIndexOffsetHelper<int64_t, 4> index_helper(y_vector.data());
//       PoolKernelUtil<device_type, T, int64_t>::Maxpool3dForward(
//           ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
//     }
//   };
// };

// template<DeviceType device_type, typename T>
// class MaxPool3dGradKernel final : public user_op::OpKernel {
//  public:
//   MaxPool3dGradKernel() = default;
//   ~MaxPool3dGradKernel() = default;

//   bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
//   std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
//       user_op::KernelCacheContext* ctx) const override {
//     return CreatePoolOpKernelCache(ctx, 3);
//   }

//  private:
//   void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
//                const user_op::OpKernelCache* cache) const override {
//     const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
//     const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
//     user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

//     const auto* pool_cache = dynamic_cast<const PoolOpKernelCache*>(cache);
//     const MaxPoolParams3D& params_3d = pool_cache->GetParams3D();

//     const int64_t elem_num = dy->shape_view().elem_cnt();
//     const T* src = dy->dptr<T>();
//     const int64_t* indice_ptr = indice->dptr<int64_t>();
//     T* dest = dx->mut_dptr<T>();

//     DimVector dy_vector(4);
//     dy_vector.at(0) = dy->shape_view().At(0) * dy->shape_view().At(1);
//     dy_vector.at(1) = dy->shape_view().At(2);
//     dy_vector.at(2) = dy->shape_view().At(3);
//     dy_vector.at(3) = dy->shape_view().At(4);

//     size_t out_bytes_size = dx->shape_view().elem_cnt() * GetSizeOfDataType(dx->data_type());
//     Memset<device_type>(ctx->stream(), dest, 0, out_bytes_size);

//     if (elem_num < GetMaxVal<int32_t>()) {
//       NdIndexOffsetHelper<int32_t, 4> index_helper(dy_vector.data());
//       PoolKernelUtil<device_type, T, int32_t>::Maxpool3dBackward(
//           ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
//     } else {
//       NdIndexOffsetHelper<int64_t, 4> index_helper(dy_vector.data());
//       PoolKernelUtil<device_type, T, int64_t>::Maxpool3dBackward(
//           ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
//     }
//   };
// };

// #define REGISTER_POOL_KERNELS(device, dtype)                                            \
//   REGISTER_USER_KERNEL("max_pool_1d")                                                   \
//       .SetCreateFn<MaxPool1dKernel<device, dtype>>()                                    \
//       .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
//                        && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
//   REGISTER_USER_KERNEL("max_pool_1d_grad")                                              \
//       .SetCreateFn<MaxPool1dGradKernel<device, dtype>>()                                \
//       .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
//                        && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
//   REGISTER_USER_KERNEL("max_pool_2d")                                                   \
//       .SetCreateFn<MaxPool2dKernel<device, dtype>>()                                    \
//       .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
//                        && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
//   REGISTER_USER_KERNEL("max_pool_2d_grad")                                              \
//       .SetCreateFn<MaxPool2dGradKernel<device, dtype>>()                                \
//       .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
//                        && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
//   REGISTER_USER_KERNEL("max_pool_3d")                                                   \
//       .SetCreateFn<MaxPool3dKernel<device, dtype>>()                                    \
//       .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
//                        && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
//   REGISTER_USER_KERNEL("max_pool_3d_grad")                                              \
//       .SetCreateFn<MaxPool3dGradKernel<device, dtype>>()                                \
//       .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
//                        && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

// #define REGISTER_POOL_WITH_DEVICE(device) \
//   REGISTER_POOL_KERNELS(device, int32_t)  \
//   REGISTER_POOL_KERNELS(device, float)    \
//   REGISTER_POOL_KERNELS(device, double)

// REGISTER_POOL_WITH_DEVICE(DeviceType::kCPU)

// #ifdef WITH_CUDA
// REGISTER_POOL_WITH_DEVICE(DeviceType::kCUDA)
// REGISTER_POOL_KERNELS(DeviceType::kCUDA, half)
// #endif

// OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_POOL_KERNEL_UTIL, (DeviceType::kCPU),
//                                  POOL_DATA_TYPE_CPU_SEQ, POOL_IDX_DATA_TYPE_SEQ);

#define REGISTER_UNPOOL_KERNELS(device, dtype)              \
  REGISTER_USER_KERNEL("max_unpool_1d")                     \
      .SetCreateFn<MaxUnpool1dKernel<device, dtype>>()      \
      .SetIsMatchedHob((user_op::HobDeviceType() == device) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

#define REGISTER_UNPOOL_WITH_DEVICE(device) \
  REGISTER_UNPOOL_KERNELS(device, int32_t)  \
  REGISTER_UNPOOL_KERNELS(device, float)    \
  REGISTER_UNPOOL_KERNELS(device, double)

REGISTER_UNPOOL_WITH_DEVICE(DeviceType::kCPU)

// #ifdef WITH_CUDA
// REGISTER_UNPOOL_WITH_DEVICE(DeviceType::kCUDA)
// REGISTER_UNPOOL_KERNELS(DeviceType::kCUDA, half)
// #endif

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNPOOL_KERNEL_UTIL, (DeviceType::kCPU),
                                 UNPOOL_DATA_TYPE_CPU_SEQ, UNPOOL_IDX_DATA_TYPE_SEQ);

}  // namespace oneflow
