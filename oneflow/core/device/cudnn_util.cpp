#ifdef WITH_CUDNN

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

cudnnDataType_t GetCudnnDataType(DataType val) {
#define MAKE_ENTRY(type_cpp, type_cudnn) \
  if (val == GetDataType<type_cpp>::val) { return type_cudnn; }
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, CUDNN_DATA_TYPE_SEQ);
#undef MAKE_ENTRY
  UNEXPECTED_RUN();
}  // namespace oneflow

CudnnTensorDesc::~CudnnTensorDesc() {
  CudaCheck(cudnnDestroyTensorDescriptor(val_));
}

CudnnTensorDesc::CudnnTensorDesc(DataType data_type, const Shape& shape) {
  CudaCheck(cudnnCreateTensorDescriptor(&val_));
  std::vector<int> stride_of_tensor(shape.NumAxes(), 1);
  CudaCheck(cudnnSetTensorNdDescriptor(
      val_, GetCudnnDataType(data_type), static_cast<int>(shape.NumAxes()),
      reinterpret_cast<const int*>(shape.dim_vec().data()),
      stride_of_tensor.data()));
}

CudnnFilterDesc::~CudnnFilterDesc() {
  CudaCheck(cudnnDestroyFilterDescriptor(val_));
}

CudnnFilterDesc::CudnnFilterDesc(DataType data_type, std::string data_format,
                                 const Shape& shape) {
  std::vector<int> stride_of_tensor(shape.NumAxes(), 1);
  cudnnTensorFormat_t cudnn_data_format;
  if (data_format == "NCHW" || data_format == "NCDHW") {
    cudnn_data_format = CUDNN_TENSOR_NCHW;
  } else if (data_format == "NHWC" || data_format == "NDHWC") {
    cudnn_data_format = CUDNN_TENSOR_NHWC;
  } else {
    UNEXPECTED_RUN();
  }
  CudaCheck(cudnnSetFilterNdDescriptor(
      val_, GetCudnnDataType(data_type), cudnn_data_format,
      static_cast<int>(shape.NumAxes()),
      reinterpret_cast<const int*>(shape.dim_vec().data())));
}

}  // namespace oneflow

#endif  // WITH_CUDNN
