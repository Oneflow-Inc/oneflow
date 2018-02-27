#ifndef ONEFLOW_CORE_KERNEL_MAX_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MAX_POOLING_KERNEL_H_

#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MaxPoolingIf : public Pooling<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingIf);
  MaxPoolingIf() = default;
  virtual ~MaxPoolingIf() = default;

 private:
  const Pooling3DKernelConf& GetPooling3DKernelConf() const override {
    return this->kernel_conf().max_pooling_3d_conf().pooling_3d_conf();
  }
#ifdef WITH_CUDA
  cudnnPoolingMode_t GetCudnnPoolingMode() override {
    return CUDNN_POOLING_MAX;
  }
#endif
};

template<DeviceType device_type, typename T>
class MaxPooling : public MaxPoolingIf<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling);
  MaxPooling() = default;
  ~MaxPooling() = default;
};

template<typename T>
class MaxPooling<DeviceType::kCPU, T>
    : public MaxPoolingIf<DeviceType::kCPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling);
  MaxPooling() = default;
  virtual ~MaxPooling() = default;

 private:
  void ForwardOnCPU(const Pooling3DCtx& pooling_ctx, const Blob* in_blob,
                    Blob* out_blob) const override;
  T ForwardInitialize() const override;
  void ForwardProcess(const T& lhs, T& rhs) const override;
  void ForwardProcess(
      const int64_t in_col, const int64_t out_col,
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>&
          in_mat,
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& out_mat)
      const override;
  void ForwardFinalize(const int64_t size, T& out) const override {}
  void ForwardFinalize(
      const int64_t size, const int64_t col,
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& out_mat)
      const override {}
  void BackwardOnCPU(const Pooling3DCtx& pooling_ctx, const Blob* out_diff_blob,
                     const Blob* out_blob, const Blob* in_blob,
                     Blob* in_diff_blob) const override;
  void BackwardProcessGrad(const T& in, const T& out, const T& out_diff,
                           const float scale, T& in_diff) const override;
  void BackwardProcessGrad(
      const int64_t out_col, const int64_t in_col, const float scale,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          out_arr,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          in_arr,
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          out_diff_arr,
      Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic>>&
          in_diff_arr) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MAX_POOLING_KERNEL_H_
