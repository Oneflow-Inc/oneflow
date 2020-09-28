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
#include "oneflow/core/common/balanced_splitter.h"
#include <stdarg.h>
#include <string.h>

namespace oneflow {

namespace user_op {

namespace {

/*
  A converter that converts coordinate of high-dimensions tensor
  and offset in one-dimension offset.
*/
template<typename IDXTYPE>
struct CoordinateOffsetConverter{
    CoordinateOffsetConverter(const ShapeView& tensorShape) :
    offset_(0), axisNum_(tensorShape.NumAxes())
    {

    }

    void setCoordinate(IDXTYPE x0, ...){
        va_list marker;
        coordinate_[0] = x0;
        va_start(marker, x0);
        FOR_RANGE(unsigned, i, 1, axisNum_){
            coordinate_[i] = va_arg(marker, IDXTYPE);
        }
        va_end(marker);
    }

    void setOffset(IDXTYPE idx){
        offset_ = idx;
    }

    IDXTYPE coordinateToIdx(){
        offset_ = 0;
        FOR_RANGE(IDXTYPE, i, 0, axisNum_){
            IDXTYPE tmp = 1;
            for (IDXTYPE j = i + 1; j < axisNum_; j++){
                tmp *= shape_[j];
            }
            offset_ += tmp*coordinate_[i];
        }

        return offset_;
    }

    void idxToCoordinate(){
        IDXTYPE tmp = offset_;
        for (IDXTYPE i = axisNum_-1; i >= 0; --i)
        {
            coordinate_[i] = tmp % shape_[i];
            tmp = (tmp - coordinate_[i])/shape_[i];
        }
    }

    void copyCoordinate(CoordinateOffsetConverter& otherConverter){
      memcpy(coordinate_, otherConverter.coordinate_, sizeof(IDXTYPE)*axisNum_);
    }

    IDXTYPE coordinate_[8];
    IDXTYPE offset_;
    int64_t axisNum_;
    int64_t shape_[8];
};

} // namespace

template<DeviceType device_type, typename IN_T, typename IDX_T>
class TorchGatherKernel final : public user_op::OpKernel {
 public:
  TorchGatherKernel() = default;
  ~TorchGatherKernel() override = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor *input_tensor = ctx->Tensor4ArgNameAndIndex("input", 0);
    const Tensor *index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
    Tensor *out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t dim = ctx->Attr<int64_t>("dim");
    const bool is_sparse_grad = ctx->Attr<bool>("sparse_grad");
    
    if (index_tensor->shape().elem_cnt() == 0) { 
      return; 
    }

    if(is_sparse_grad){
      //TO DO...
    }

    const IN_T* input = input_tensor->dptr<IN_T>();
    IN_T* output = out_tensor->mut_dptr<IN_T>();

    CoordinateOffsetConverter<IDX_T> input_nd_helper(input_tensor->shape());
    CoordinateOffsetConverter<IDX_T> index_nd_helper(input_tensor->shape());
    FOR_RANGE(IDX_T, index_offset, 0, input_tensor->shape().elem_cnt()){
      
      /*
        when dim = 1 
        output[i][j][k] = input[i][x][k] 
        where x is:
          x = index[i][j][k]
      */

      // get i, j, k
      index_nd_helper.setOffset(index_offset);
      index_nd_helper.idxToCoordinate();

      // re-write x at axis "dim", updates offset    
      const IDX_T* index = index_tensor->dptr<IDX_T>();
      const IDX_T x = index[index_offset];
      input_nd_helper.copyCoordinate(index_nd_helper);
      input_nd_helper.coordinate_[dim] = x;
      IDX_T input_offset = input_nd_helper.coordinateToIdx();    

      output[index_offset] = input[input_offset];
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_PYTORCH_GATHER_KERNEL(device, in_type, indices_type)                                \
  REGISTER_USER_KERNEL("torch_gather")                                                             \
      .SetCreateFn<                                                                          \
          TorchGatherKernel<device, OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indices_type)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                   \
                       & (user_op::HobDataType("input", 0) == OF_PP_PAIR_SECOND(in_type))       \
                       & (user_op::HobDataType("index", 0) == OF_PP_PAIR_SECOND(indices_type)));

#define GATHER_DATA_TYPE_SEQ ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ


OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_PYTORCH_GATHER_KERNEL,
                                (DeviceType::kCPU), 
                                GATHER_DATA_TYPE_SEQ,
                                INDEX_DATA_TYPE_SEQ)

}  // namespace user_op

}  // namespace oneflow
