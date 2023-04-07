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

#ifndef ONEFLOW_USER_KERNELS_CUFFT_PLAN_CACHE_H_
#define ONEFLOW_USER_KERNELS_CUFFT_PLAN_CACHE_H_

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

constexpr int max_rank = 3;

enum class CUFFT_EXCUTETYPE{ R2C, C2C, C2R };

struct CuFFT_DType_Desc{
  cudaDataType inputtype;
  cudaDataType outputtype;
  cudaDataType executiontype;
};

}


// NOTE: The implementation of `_cudaGetErrorEnum`  are mostly taken from
// pytorch.
//       For more details pls refer to:
//       https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/CuFFTUtils.h#L17
static inline std::string _cudaGetErrorEnum(cufftResult error)
{
  switch (error)
  {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
    default:
      std::ostringstream ss;
      ss << "unknown error " << error;
      return ss.str();
  }
}

static inline void CUFFT_CHECK(cufftResult error)
{
  CHECK_OR_THROW(error == CUFFT_SUCCESS) << "cuFFT error: " << _cudaGetErrorEnum(error);
}

class CuFFTHandle{
  cufftHandle handle;
public:
  CuFFTHandle(){
    CUFFT_CHECK(cufftCreate(&handle));
  }

  cufftHandle* get(){
    return &handle;
  }
  const cufftHandle* get() const{
    return &handle;
  }

  ~CuFFTHandle(){
    cufftDestroy(handle);
  }
};

struct CuFFtParams {
  int32_t ndim;
  int32_t output_shape[max_rank + 1];
  int32_t input_shape[max_rank + 1];
  int32_t input_strides[max_rank + 1];
  int32_t output_strides[max_rank + 1];
  bool IsForward;
  CUFFT_EXCUTETYPE excute_type;
  DataType real_data_type;

  // int32_t* rank;
  // int32_t batch = 0;

  CuFFtParams() = default;
  CuFFtParams(const Shape& in_shape, const Shape& out_shape, const Stride& in_strides,
              const Stride& out_strides, int32_t dims, const bool is_forward,
              CUFFT_EXCUTETYPE type, DataType real) : ndim(dims), IsForward(is_forward), excute_type(type), real_data_type(real)
              {
        assert(ndim >= 1 && ndim <= max_rank);
        assert(in_shape.size() == in_stride.size());
        assert(out_shape.size() == out_stride.size());

        std::copy(in_strides.begin(), in_strides.end(), input_strides);
        std::copy(out_strides.begin(), out_strides.end(), output_strides);
        std::copy(in_shape.begin(), in_shape.end(), input_shape);
        std::copy(out_shape.begin(), out_shape.end(), output_shape);
  }
};

template<typename IN, typename OUT>
class CuFFtConfig {
 public:
  CuFFtConfig(const CuFFtConfig&) = delete;
  CuFFtConfig& operator=(CuFFtConfig const&) = delete;
  ~CuFFtConfig() = default;

  explicit CuFFtConfig(CuFFtParams& params) {  // NOLINT
    // cufftPlanMany(&plan_handle_, params.ndim, params.rank, params.input_shape,
    //               params.input_strides[0], params.input_strides[1], params.output_shape,
    //               params.output_strides[0], params.output_strides[1], exectype_, params.batch);

    if (params.real_data_type == kBFloat16 || params.real_data_type == kFloat16){
      // CuFFT support half data type, but there are some limits:
      //  https://docs.nvidia.com/cuda/cufft/#half-precision-cufft-transforms
      // TO-DO : do some check
    }
    

    infer_cufft_type_(params.excute_type, params.real_data_type);

    cufftXtMakePlanMany(&plan_handle_, params.ndim, params.input_shape, params.input_shape, 
          params.input_strides[0], long long idist, cudaDataType inputtype, 
          long long *onembed, long long ostride, long long odist, 
          cudaDataType outputtype, long long batch, size_t *workSize, 
          cudaDataType executiontype)
  }


 private:
  void infer_cufft_type_(CUFFT_EXCUTETYPE excute_type, DataType real_data_type) {
    if (real_data_type == kFloat16){
      data_type_desc.executiontype = CUDA_C_16F;
      data_type_desc.inputtype = excute_type == CUFFT_EXCUTETYPE::R2C ? CUDA_R_16F : CUDA_C_16F;
      data_type_desc.outputtype = excute_type == CUFFT_EXCUTETYPE::C2R ? CUDA_R_16F : CUDA_C_16F;
    }
    else if (real_data_type == kBFloat16){
      data_type_desc.executiontype = CUDA_C_16BF;
      data_type_desc.inputtype = excute_type == CUFFT_EXCUTETYPE::R2C ? CUDA_R_16BF : CUDA_C_16BF;
      data_type_desc.outputtype = excute_type == CUFFT_EXCUTETYPE::C2R ? CUDA_R_16BF : CUDA_C_16BF;
    }
    else if (real_data_type == kFloat){
      data_type_desc.executiontype = CUDA_C_32F;
      data_type_desc.inputtype = excute_type == CUFFT_EXCUTETYPE::R2C ? CUDA_R_32F : CUDA_C_32F;
      data_type_desc.outputtype = excute_type == CUFFT_EXCUTETYPE::C2R ? CUDA_R_32F : CUDA_C_32F;
    }
    else if (real_data_type == kDouble){
      data_type_desc.executiontype = CUDA_C_64F;
      data_type_desc.inputtype = excute_type == CUFFT_EXCUTETYPE::R2C ? CUDA_R_64F : CUDA_C_64F;
      data_type_desc.outputtype = excute_type == CUFFT_EXCUTETYPE::C2R ? CUDA_R_64F : CUDA_C_64F;
    }
    else{
      CHECK_OR_THROW(false) << "cuFFT doesn't support type " << real_data_type;
    }
  }

  CuFFTHandle plan_handle_;
  // cufftType cufft_exectype_;
  CuFFT_DType_Desc data_type_desc;
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_CUFFT_PLAN_CACHE_H_
