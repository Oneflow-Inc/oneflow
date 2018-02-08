#ifndef ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

#ifdef WITH_CUDA
class CudnnPoolingNdDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnPoolingNdDesc);
  CudnnPoolingNdDesc() = delete;
  ~CudnnPoolingNdDesc();

  CudnnPoolingNdDesc(PoolingMode pooling_mode, const std::vector<int>& window,
                     const std::vector<int>& padding,
                     const std::vector<int>& stride);

  const cudnnPoolingDescriptor_t& Get() const { return val_; }

 private:
  cudnnPoolingDescriptor_t val_;
};
#endif

class Pooling3DCtx {
 public:
  Pooling3DCtx() = default;
  ~Pooling3DCtx();

#define DEFINE_POOLING_DHW_SETTER_AND_GETTER(name)                \
  int32_t name##_d() const { return name##_d_; }                  \
  int32_t name##_h() const { return name##_h_; }                  \
  int32_t name##_w() const { return name##_w_; }                  \
  void set_##name##_d(int32_t name##_d) { name##_d_ = name##_d; } \
  void set_##name##_h(int32_t name##_h) { name##_h_ = name##_h; } \
  void set_##name##_w(int32_t name##_w) { name##_w_ = name##_w; }

  DEFINE_POOLING_DHW_SETTER_AND_GETTER(pool_size);
  DEFINE_POOLING_DHW_SETTER_AND_GETTER(strides);
  DEFINE_POOLING_DHW_SETTER_AND_GETTER(padding);
  DEFINE_POOLING_DHW_SETTER_AND_GETTER(in);
  DEFINE_POOLING_DHW_SETTER_AND_GETTER(out);

#undef DEFINE_POOLING_DHW_SETTER_AND_GETTER

#define DEFINE_POOLING_NC_SETTER_AND_GETTER(name)                 \
  int32_t name##_n() const { return name##_n_; }                  \
  int32_t name##_c() const { return name##_c_; }                  \
  void set_##name##_n(int32_t name##_n) { name##_n_ = name##_n; } \
  void set_##name##_c(int32_t name##_c) { name##_c_ = name##_c; }

  DEFINE_POOLING_NC_SETTER_AND_GETTER(in);
  DEFINE_POOLING_NC_SETTER_AND_GETTER(out);

#undef DEFINE_POOLING_NC_SETTER_AND_GETTER

  void BuildCudnnDescs(PoolingMode mode, DataType type);
#ifdef WITH_CUDA
  CudnnTensorNdDesc* in_desc_ptr() const { return in_desc_; }
  CudnnTensorNdDesc* in_diff_desc_ptr() const { return in_diff_desc_; }
  CudnnTensorNdDesc* out_desc_ptr() const { return out_desc_; }
  CudnnTensorNdDesc* out_diff_desc_ptr() const { return out_diff_desc_; }
  CudnnPoolingNdDesc* pooling_desc_ptr() const { return pooling_desc_; }
#endif  // WITH_CUDA

 private:
  int32_t in_n_;
  int32_t in_c_;
  int32_t in_d_;
  int32_t in_h_;
  int32_t in_w_;
  int32_t pool_size_d_;
  int32_t pool_size_h_;
  int32_t pool_size_w_;
  int32_t strides_d_;
  int32_t strides_h_;
  int32_t strides_w_;
  int32_t padding_d_;
  int32_t padding_h_;
  int32_t padding_w_;
  int32_t out_n_;
  int32_t out_c_;
  int32_t out_d_;
  int32_t out_h_;
  int32_t out_w_;

#ifdef WITH_CUDA
  CudnnTensorNdDesc* in_desc_;
  CudnnTensorNdDesc* in_diff_desc_;
  CudnnTensorNdDesc* out_desc_;
  CudnnTensorNdDesc* out_diff_desc_;
  CudnnPoolingNdDesc* pooling_desc_;
#endif  // WITH_CUDA
};

template<DeviceType device_type>
class PoolingKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernel);
  PoolingKernel() = default;
  virtual ~PoolingKernel() = default;

 protected:
  virtual PoolingMode GetPoolingMode() = 0;
  const Pooling3DCtx& pooling_3d_ctx() const { return pooling_3d_ctx_; }
  Pooling3DCtx* mut_pooling_3d_ctx() { return &pooling_3d_ctx_; }

  Pooling3DCtx pooling_3d_ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
