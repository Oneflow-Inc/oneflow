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
#include "oneflow/opencl/ep/primitive/broadcast_elementwise_binary.h"
#include "oneflow/opencl/ep/primitive/type_seq.h"
#include "oneflow/core/ep/common/primitive/broadcast_elementwise_binary.h"
#include "oneflow/opencl/common/cl_api.h"
#include "oneflow/opencl/ep/cl_stream.h"

namespace oneflow {
namespace ep {
namespace primitive {
namespace opencl {

template<BinaryOp op, typename T>
class BinaryMath : public BroadcastElementwiseBinary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BinaryMath);
  BinaryMath() {
    OF_CL_CHECK(
        clBuildKernel("cl_binary", "cl_binary", &kernel,
                      "-DOP=" + getBinaryOpAsString<op>() + " -DDT=" + getCppDTypeAsString<T>()));
  }

  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              size_t num_src1_dims, const int64_t* src1_dims, const void* src1, void* dst) {
    CHECK_EQ_OR_THROW(num_src0_dims, num_src1_dims)
        << "broadcasting binary operation is not supported!";
    for (int i = 0; i < num_src0_dims; ++i) {
      CHECK_EQ_OR_THROW(src0_dims[i], src1_dims[i])
          << "broadcasting binary operation is not supported!";
    }
    std::vector<int64_t> dst_dims;
    if (num_src0_dims > num_src1_dims) {
      dst_dims = ComputeBroadcastShape(num_src1_dims, src1_dims, num_src0_dims, src0_dims);
    } else {
      dst_dims = ComputeBroadcastShape(num_src0_dims, src0_dims, num_src1_dims, src1_dims);
    }
    int64_t count = ComputeElementCount(dst_dims.size(), dst_dims.data());
    int index = 0;
    kernel.setArg(index++, static_cast<int>(count));
    kernel.setArg(index++, *reinterpret_cast<const cl::Buffer*>(src0));
    kernel.setArg(index++, *reinterpret_cast<const cl::Buffer*>(src1));
    kernel.setArg(index++, *reinterpret_cast<cl::Buffer*>(dst));

    cl::NDRange global_worksize{static_cast<size_t>((count + 7) >> 3)};
    OF_CL_CHECK(clLaunchKernel(kernel, cl::NullRange, global_worksize, cl::NullRange,
                               stream->As<clStream>()->cl_stream()));
  }

  void Launch(Stream* stream, Scalar src0, size_t num_src1_dims, const int64_t* src1_dims,
              const void* src1, void* dst) {
    THROW(RuntimeError) << "scalar binary operation is not supported!";
  }

  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              Scalar src1, void* dst) {
    THROW(RuntimeError) << "scalar binary operation is not supported!";
  }

 private:
  cl::Kernel kernel;
};

#define INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY(binary_op, src_data_type_pair) \
  template<>                                                                                   \
  std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary<                   \
      binary_op, OF_PP_PAIR_FIRST(src_data_type_pair), OF_PP_PAIR_FIRST(src_data_type_pair)>(  \
      Scalar attr0, Scalar attr1) {                                                            \
    return std::unique_ptr<BroadcastElementwiseBinary>(                                        \
        new BinaryMath<binary_op, OF_PP_PAIR_FIRST(src_data_type_pair)>);                      \
  }

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY,
                                 CL_BINARY_MATH_OP_SEQ, CL_PRIMITIVE_ALL_TYPE_SEQ);

#undef INSTANTIATE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY

}  // namespace opencl
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
