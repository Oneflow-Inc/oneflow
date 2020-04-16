#ifndef ONEFLOW_CUSTOMIZED_UTILS_POOL_UTIL_H_
#define ONEFLOW_CUSTOMIZED_UTILS_POOL_UTIL_H_
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/eigen_util.h"
#include "oneflow/customized/kernels/op_kernel_state_wrapper.h"

namespace oneflow {

typedef fixed_vector<int64_t, SHAPE_MAX_AXIS_SIZE> FixedDimVector;
typedef fixed_vector<int32_t, SHAPE_MAX_AXIS_SIZE> FixedVector;

class Params3D {
 public:
  Params3D(const int32_t dim, const Shape& x_shape, const std::string& data_format,
           const std::string& padding, const std::vector<int32_t>& pool_size,
           const std::vector<int32_t>& strides);
  ~Params3D() = default;

  Shape GetYShape() const;
  Shape GetXShape5D() const;
  Shape GetYShape5D() const;

  const std::vector<int32_t>& pool_size_3d() const { return pool_size_3d_; }
  const std::vector<int32_t>& strides_3d() const { return strides_3d_; }
  const std::vector<int32_t>& padding_before_3d() const { return padding_before_3d_; }
  const std::vector<int32_t>& padding_after_3d() const { return padding_after_3d_; }

 private:
  int32_t dim_;
  FixedDimVector x_3d_;
  FixedDimVector y_3d_;
  std::vector<int32_t> pool_size_3d_;
  std::vector<int32_t> strides_3d_;
  std::vector<int32_t> padding_before_3d_;
  std::vector<int32_t> padding_after_3d_;
  std::string data_format_;
  int64_t batch_num_;
  int64_t channel_num_;
};

class CudnnPoolDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnPoolDesc);
  CudnnPoolDesc() = delete;
  ~CudnnPoolDesc();

  CudnnPoolDesc(cudnnPoolingMode_t pooling_mode, int dims, const int* window, const int* padding,
                const int* stride);

  const cudnnPoolingDescriptor_t& Get() const { return val_; }

 private:
  cudnnPoolingDescriptor_t val_;
};

class GPUPoolOpKernelState final {
 public:
  GPUPoolOpKernelState(const int32_t dim, const std::string& pooling_type, const Shape& x_shape,
                       const Shape& y_shape, const std::string& data_format, const DataType& dtype,
                       const Params3D& params_3d);
  ~GPUPoolOpKernelState() = default;

  static std::shared_ptr<user_op::OpKernelState> FromKernelInitContext(
      const int32_t& dim, const std::string& pooling_type, user_op::KernelInitContext* ctx);
  const cudnnTensorDescriptor_t& cudnn_x_tensor_desc() const { return x_desc_->Get(); }
  const cudnnTensorDescriptor_t& cudnn_y_tensor_desc() const { return y_desc_->Get(); }
  const cudnnPoolingDescriptor_t& cudnn_pooling_desc() const { return pooling_desc_->Get(); }

 private:
  std::unique_ptr<CudnnTensorDesc> x_desc_;
  std::unique_ptr<CudnnTensorDesc> y_desc_;
  std::unique_ptr<CudnnPoolDesc> pooling_desc_;
};

template<typename T>
struct PoolKernelUtil {
 public:
  typedef std::function<T()> ForwardInitialize;
  typedef std::function<void(const T& lhs, T& rhs)> CFirstProcess;
  typedef std::function<void(const int64_t in_col, const int64_t out_col,
                             ConstEigenMatrixMap<T>& in_mat, EigenMatrixMap<T>& out_mat)>
      CLastProcess;
  typedef std::function<void(const int64_t size, T& out)> CFirstFinalize;
  typedef std::function<void(const int64_t size, const int64_t col, EigenMatrixMap<T>& out_mat)>
      CLastFinalize;
  typedef std::function<void(const T& in, const T& out, const T& out_diff, const int64_t size,
                             T& in_diff)>
      CFirstProcessGrad;
  typedef std::function<void(const int64_t out_col, const int64_t in_col, const int64_t size,
                             ConstEigenArrayMap<T>& out_arr, ConstEigenArrayMap<T>& in_arr,
                             ConstEigenArrayMap<T>& out_diff_arr, EigenArrayMap<T>& in_diff_arr)>
      CLastProcessGrad;

  static void CFirstForward(const Params3D& params_3d, const user_op::Tensor* in_blob,
                            user_op::Tensor* out_blob, const ForwardInitialize& initialize,
                            const CFirstProcess& process, const CFirstFinalize& finalize);
  static void CFirstBackward(const Params3D& params_3d, const user_op::Tensor* out_diff_blob,
                             const user_op::Tensor* out_blob, const user_op::Tensor* in_blob,
                             user_op::Tensor* in_diff_blob, const CFirstProcessGrad& process);
  static void CLastForward(const Params3D& params_3d, const user_op::Tensor* in_blob,
                           user_op::Tensor* out_blob, const ForwardInitialize& forward_initialize,
                           const CLastProcess& process, const CLastFinalize& finalize);
  static void CLastBackward(const Params3D& params_3d, const user_op::Tensor* out_diff_blob,
                            const user_op::Tensor* out_blob, const user_op::Tensor* in_blob,
                            user_op::Tensor* in_diff_blob, const CLastProcessGrad& process);
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_UTILS_POOL_UTIL_H_
