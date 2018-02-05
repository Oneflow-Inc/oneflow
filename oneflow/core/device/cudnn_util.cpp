#ifdef WITH_CUDNN

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

CudnnTensorDesc::CudnnTensorDesc(DataType data_type, int n, int c, int h,
                                 int w) {
  Init(data_type, n, c, h, w);
}

CudnnTensorDesc::CudnnTensorDesc(DataType data_type, const Shape& shape) {
  CHECK_EQ(shape.NumAxes(), 4);
  Init(data_type, shape.At(0), shape.At(1), shape.At(2), shape.At(3));
}

void CudnnTensorDesc::Init(DataType data_type, int n, int c, int h, int w) {
  CudaCheck(cudnnCreateTensorDescriptor(&val_));
  CudaCheck(cudnnSetTensor4dDescriptor(
      val_, CUDNN_TENSOR_NCHW, GetCudnnDataType(data_type), n, c, h, w));
}

CudnnFilterDesc::~CudnnFilterDesc() {
  CudaCheck(cudnnDestroyFilterDescriptor(val_));
}

CudnnFilterDesc::CudnnFilterDesc(DataType data_type, int k, int c, int h,
                                 int w) {
  Init(data_type, k, c, h, w);
}

CudnnFilterDesc::CudnnFilterDesc(DataType data_type, const Shape& shape) {
  CHECK_EQ(shape.NumAxes(), 4);
  Init(data_type, shape.At(0), shape.At(1), shape.At(2), shape.At(3));
}

void CudnnFilterDesc::Init(DataType data_type, int k, int c, int h, int w) {
  CudaCheck(cudnnCreateFilterDescriptor(&val_));
  CudaCheck(cudnnSetFilter4dDescriptor(val_, GetCudnnDataType(data_type),
                                       CUDNN_TENSOR_NCHW, k, c, h, w));
}

}  // namespace oneflow

#endif  // WITH_CUDNN
