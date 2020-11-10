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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/memory_copier.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow{

namespace{

/**
 * @brief Find the dimension factorial \n 
 * such as shape=(2,3,4,5), idx=0, then num = 5*4*3*2=120; idx=1, then num=5*4*3=60
 */
int64_t dim_factorial(int64_t idx, const std::vector<int64_t> shape){
    int64_t num = 1;
    for(idx; idx < shape.size(); idx++){
        num *= shape[idx];
    }
    return num;
}


/**
 * @brief Convert high-dimensional array coordinates to one-dimensional array index
 */
int64_t coordinate_to_index(const std::vector<int64_t> coordinate,  const std::vector<int64_t> shape, const int64_t size_of_data_type){
    int64_t idx = 0;
    for(int64_t i = 0; i<shape.size(); ++i){
        idx += coordinate[i] * dim_factorial(i+1, shape);
    }
    return idx;
}


/**
 * @brief Convert high-dimensional array coordinates to one-dimensional array index
 */
int64_t coordinate_to_index(const std::vector<int64_t> coordinate,  const ShapeView shape, const int64_t size_of_data_type){
    int64_t idx = 0;
    for(int64_t i = 0; i<shape.NumAxes(); ++i){
        idx += coordinate[i] * shape.Count(i+1);
    }
    return idx;
}


/**
 * @brief Convert one-dimensional array index to high-dimensional coordinates
 */
std::vector<int64_t>  index_to_coordinate(const int64_t idx, const std::vector<int64_t> shape, std::vector<int64_t> &  coordinate){
  int64_t  tmp = idx;
  int64_t i = shape.size()-1;
  while(i >=0){
      int64_t dim_i_idx = (tmp % shape[i]);
      coordinate[i] =  dim_i_idx;
      tmp = (tmp-dim_i_idx)/shape[i];
      i -= 1;
  }
  return coordinate;
}


/**
 * @brief Get dim vector with size of data type.\n
 * Replace the last dimension of the dim vector with: the number of elements n × the number of bytes of the data type \n
 *  (if there are 4 float32 types in the last dimension, then 4×4=16)
 */
void GetDimVectorInBytes(const ShapeView& tensor_shape, const int64_t size_of_data_type,
                         DimVector& shape_vec) {
   int64_t ndims = tensor_shape.NumAxes();
  for (int64_t i = 0; i < ndims; ++i) {
    shape_vec[i] = tensor_shape.At(i); 
  }
  shape_vec[ndims - 1] = shape_vec[ndims - 1] * size_of_data_type;
}


/**
 * @brief Fill ShapeView into vector
 */
std::vector<int64_t>  shapeview_to_vector(const ShapeView& tensor_shape) {
    int64_t ndims = tensor_shape.NumAxes();
    std::vector<int64_t> shape_vec(ndims);
    for (int64_t i = 0; i < ndims; ++i) {
      shape_vec[i] = tensor_shape.At(i); 
    }
    shape_vec[ndims - 1] = shape_vec[ndims - 1];
    return shape_vec;
}


/**
 * @brief Fill ShapeView into dim vector
 */
DimVector  shapeview_to_dimvector(const ShapeView& tensor_shape) {
    int64_t ndims = tensor_shape.NumAxes();
    DimVector shape_vec(ndims);
    for (int64_t i = 0; i < ndims; ++i) {
      shape_vec[i] = tensor_shape.At(i); 
    }
    shape_vec[ndims - 1] = shape_vec[ndims - 1];
    return shape_vec;
}

} //namespace


template<DeviceType device_type, typename T>
class ReflectionPad2dKernel final : public user_op::OpKernel {

 public:
  ReflectionPad2dKernel() = default;
  ~ReflectionPad2dKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
      user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
      const auto& padding= ctx->Attr<std::vector<int64_t>>("padding");
      const std::string  data_format = ctx->Attr<std::string>("data_format");
      const int64_t ndims = x->shape().NumAxes();
      const int64_t sizeof_dtype = static_cast<int64_t>(GetSizeOfDataType(x->data_type()));

      CHECK_EQ(padding.size(), ndims);
    //   NewKernelUtil<device_type>::Fill(ctx->device_ctx(), y->shape().elem_cnt(),
    //                                  static_cast<T>(0.0), y->mut_dptr<T>());

      MemoryCopyNdDesc memory_copy_nd_desc;
      DimVector src_shape_vec(ndims);
      DimVector dst_shape_vec(ndims);
      DimVector  x_vector = shapeview_to_dimvector(x->shape());
      DimVector  y_vector = shapeview_to_dimvector(y->shape());

      GetDimVectorInBytes(x->shape(), sizeof_dtype, src_shape_vec);
      GetDimVectorInBytes(y->shape(), sizeof_dtype, dst_shape_vec);
      memory_copy_nd_desc.src_shape = Shape(src_shape_vec);
      memory_copy_nd_desc.dst_shape = Shape(dst_shape_vec);

      DimVector src_pos_vec(ndims, 0);
      DimVector dst_pos_vec(padding.cbegin(), padding.cend());
      dst_pos_vec[ndims - 1] *= sizeof_dtype;
      memory_copy_nd_desc.dst_pos = NdIndex(dst_pos_vec);
      memory_copy_nd_desc.src_pos = NdIndex(src_pos_vec);
      memory_copy_nd_desc.extent = memory_copy_nd_desc.src_shape;
      MemoryCopyNdDesc reduced_memory_copy_nd_desc = memory_copy_nd_desc.CreateDimReducedDesc();
      std::unique_ptr<MemoryCopier> device_memory_copier(NewDefaultMemoryCopier(device_type));
      device_memory_copier->Copy(ctx->device_ctx(), y->mut_dptr<T>(), x->dptr<T>(),
                               reduced_memory_copy_nd_desc);


    int64_t padding_h, padding_w, channel_h_idx, channel_w_idx;
    if (data_format == "NCHW"){
        channel_h_idx = 2;
        channel_w_idx = 3;
    }else{
        channel_h_idx = 1;
        channel_w_idx = 2;
    }
    padding_h = padding[channel_h_idx];
    padding_w = padding[channel_w_idx];
    const ShapeView&  x_shape = x->shape();
    const ShapeView&  y_shape = y->shape();
    //elements index vector of diagonal elements
    std::vector<int64_t> index_vector; 
    
    int  y_vector_count  = y->shape().elem_cnt();
    for(int i = 0;i< y_vector_count; i++){
        //Traverse one-dimensional array y
        std::vector<int64_t>  coordinate(y_shape.NumAxes());
        std::vector<int64_t>  coord_y = index_to_coordinate(i, shapeview_to_vector(y_shape), coordinate);
        int channel_h;
        int64_t x_h = coord_y[channel_h_idx] - padding_h;
        int64_t  x_w = coord_y[channel_w_idx] - padding_w;
        //printf("i:%d >>>>>>coord_y:  [%ld, %ld, %ld, %ld];  coord_x: [%ld, %ld, %ld, %ld]\n", i, coord_y[0], coord_y[1], coord_y[2], coord_y[3], coord_y[0], coord_y[1], x_h, x_w);
        if(x_h < 0 || x_h >= x_shape.At(channel_h_idx) || x_w < 0 || x_w >= x_shape.At(channel_w_idx)){
            //Indicates that the element is no longer in the original x range (the data to be padding outside)
            std::vector<int64_t> dest_coords;
            int64_t dest_index;
            //Determine whether it is a diagonal element
            if((x_h >= 0 && x_h < x_shape.At(2))){
                //Within the left and right range lines, non-diagonal elements
                int64_t dest_w;
                if(x_w < 0){
                    //left part
                    dest_w = 2*padding_w - coord_y[channel_w_idx];
                }else{ 
                    //rithr pary
                    dest_w = 2*(padding_w + x_shape.At(channel_w_idx) - 1) - coord_y[channel_w_idx];
                }
                if(data_format == "NCHW"){
                    dest_coords = {coord_y[0], coord_y[1], coord_y[2], dest_w};
                }else{
                    dest_coords = {coord_y[0], coord_y[1], dest_w, coord_y[3]};
                }
                dest_index = coordinate_to_index(dest_coords, y_shape, 1);
                Memcpy<device_type>(ctx->device_ctx(), y->mut_dptr<T>() + i, y->mut_dptr<T>() + dest_index, sizeof_dtype);
            }else if( x_w >= 0 && x_w < x_shape.At(channel_w_idx)){
                //Within the upper and lower range lines, non-diagonal elements
                int64_t dest_h;
                if(x_h < 0){
                    //upper part 
                    dest_h = 2*padding_h-coord_y[channel_h_idx];
                }else{
                    //lower part
                    dest_h = 2*(padding_h + x_shape.At(channel_h_idx)-1) - coord_y[channel_h_idx];
                }
                if(data_format == "NCHW"){
                    dest_coords = {coord_y[0], coord_y[1], dest_h, coord_y[3]};
                }else{
                    dest_coords = {coord_y[0], dest_h, coord_y[2], coord_y[3]};
                }
                dest_index = coordinate_to_index(dest_coords, y_shape, 1);
                Memcpy<device_type>(ctx->device_ctx(), y->mut_dptr<T>() + i, y->mut_dptr<T>() + dest_index, sizeof_dtype);
            }else{
                //Diagonal element
                index_vector.push_back(i);           
            }
        }
     }


     //Traverse the diagonal elements around index_vector and assign values
    for(int i=0; i<index_vector.size(); i++){
        std::vector<int64_t>  coordinate(y_shape.NumAxes());
        std::vector<int64_t>  coord_y = index_to_coordinate(index_vector[i], shapeview_to_vector(y_shape), coordinate);
        int64_t dest_w;
        int64_t dest_index;
        std::vector<int64_t> dest_coords;
        if(coord_y[channel_w_idx] <  padding_w){
            //left part
            dest_w = 2 * padding_w-coord_y[channel_w_idx];
        }else{
            //right part
            dest_w = 2 * (padding_w + x_shape.At(channel_w_idx) - 1) - coord_y[channel_w_idx];
        }
        if(data_format == "NCHW"){
            dest_coords = {coord_y[0], coord_y[1], coord_y[2], dest_w};
        }else{
            dest_coords = {coord_y[0], coord_y[1], dest_w, coord_y[3]};
        }
        dest_index = coordinate_to_index(dest_coords, y_shape, 1);
        Memcpy<device_type>(ctx->device_ctx(), y->mut_dptr<T>() + index_vector[i], y->mut_dptr<T>() + dest_index, sizeof_dtype);
    }

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REFLECTION_PAD2D_KERNELS(dev, dtype)                                             \
  REGISTER_USER_KERNEL("reflection_pad2d").SetCreateFn<ReflectionPad2dKernel<dev, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == dev)                                              \
      & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));


#ifdef WITH_CUDA
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, float)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, double)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, float16)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, int32_t)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, int64_t)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, int8_t)
#endif
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, float)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, double)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, int32_t)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, int64_t)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, int8_t)


template<DeviceType device_type, typename T>
class ReflectionPad2dGradKernel final : public user_op::OpKernel {

    public:
        ReflectionPad2dGradKernel() = default;
        ~ReflectionPad2dGradKernel() = default;
    private:
        void Compute(user_op::KernelComputeContext* ctx) const override {
            const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
            user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
            const auto& padding= ctx->Attr<std::vector<int64_t>>("padding");
            const std::string  data_format = ctx->Attr<std::string>("data_format");
            const int64_t ndims = dy->shape().NumAxes();
            const int64_t size_of_data_type = static_cast<int64_t>(GetSizeOfDataType(dy->data_type()));

            CHECK_EQ(padding.size(), ndims);


            // printf("padding >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>[%ld, %ld, %ld, %ld]\n", padding[0], padding[1], padding[2], padding[3]);
            // printf("x.shape; y.shape >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>(%ld,%ld,%ld,%ld);(%ld,%ld,%ld,%ld)\n", x->shape().At(0),  x->shape().At(1),  x->shape().At(2), 
            // x->shape().At(3),y->shape().At(0),  y->shape().At(1),  y->shape().At(2),  y->shape().At(3));


            MemoryCopyNdDesc memory_copy_nd_desc;
            DimVector src_shape_vec(ndims);
            DimVector dst_shape_vec(ndims);


            DimVector  x_vector = shapeview_to_dimvector(dy->shape());
            DimVector  y_vector = shapeview_to_dimvector(dx->shape());
            

            GetDimVectorInBytes(dy->shape(), size_of_data_type, src_shape_vec);
            GetDimVectorInBytes(dx->shape(), size_of_data_type, dst_shape_vec);


            memory_copy_nd_desc.src_shape = Shape(src_shape_vec);
            memory_copy_nd_desc.dst_shape = Shape(dst_shape_vec);


            DimVector src_pos_vec(ndims, 0);
            DimVector dst_pos_vec(padding.cbegin(), padding.cend());
            dst_pos_vec[ndims - 1] *= size_of_data_type;

            memory_copy_nd_desc.dst_pos = NdIndex(dst_pos_vec);
            memory_copy_nd_desc.src_pos = NdIndex(src_pos_vec);
            memory_copy_nd_desc.extent = memory_copy_nd_desc.src_shape;
            MemoryCopyNdDesc reduced_memory_copy_nd_desc = memory_copy_nd_desc.CreateDimReducedDesc();
            std::unique_ptr<MemoryCopier> device_memory_copier(NewDefaultMemoryCopier(device_type));
            device_memory_copier->Copy(ctx->device_ctx(), dx->mut_dptr<T>(), dy->dptr<T>(),
                                    reduced_memory_copy_nd_desc);

        }
        bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
    

#define REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(dev, dtype)            \
  REGISTER_USER_KERNEL("reflection_pad2d_grad")                      \
      .SetCreateFn<ReflectionPad2dGradKernel<dev, dtype>>()         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == dev) \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

#ifdef WITH_CUDA
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, float)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, double)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, float16)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, int32_t)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, int64_t)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, int8_t)
#endif
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, float)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, double)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, int32_t)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, int64_t)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, int8_t)


}  // namespace oneflow