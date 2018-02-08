#ifndef ONEFLOW_CORE_KERNEL_POOLING_3D_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_3D_KERNEL_H_

#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class Pooling3DKernel : public PoolingKernel<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling3DKernel);
  Pooling3DKernel() = default;
  virtual ~Pooling3DKernel() = default;

 protected:
  void VirtualKernelInit(const ParallelContext*) override {
    Pooling3DCtx* pooling_3d_ctx = this->mut_pooling_3d_ctx();
    const PbMessage& op_conf = GetPooling3DOpConf();
    const PbRf<int32_t>& pool_size =
        GetPbRfFromPbMessage<int32_t>(op_conf, "pool_size");
    pooling_3d_ctx->set_pool_size_d(pool_size.Get(0));
    pooling_3d_ctx->set_pool_size_h(pool_size.Get(1));
    pooling_3d_ctx->set_pool_size_w(pool_size.Get(2));
    const PbRf<int32_t>& strides =
        GetPbRfFromPbMessage<int32_t>(op_conf, "strides");
    pooling_3d_ctx->set_strides_d(strides.Get(0));
    pooling_3d_ctx->set_strides_h(strides.Get(1));
    pooling_3d_ctx->set_strides_w(strides.Get(2));

    const Pooling3DKernelConf& kernel_conf = GetPooling3DKernelConf();
    pooling_3d_ctx->set_padding_d(kernel_conf.padding_d());
    pooling_3d_ctx->set_padding_h(kernel_conf.padding_h());
    pooling_3d_ctx->set_padding_w(kernel_conf.padding_w());

    pooling_3d_ctx->set_in_n(kernel_conf.in_shape(0));
    pooling_3d_ctx->set_in_c(kernel_conf.in_shape(1));
    pooling_3d_ctx->set_in_d(kernel_conf.in_shape(2));
    pooling_3d_ctx->set_in_h(kernel_conf.in_shape(3));
    pooling_3d_ctx->set_in_w(kernel_conf.in_shape(4));

    pooling_3d_ctx->set_out_n(kernel_conf.out_shape(0));
    pooling_3d_ctx->set_out_c(kernel_conf.out_shape(1));
    pooling_3d_ctx->set_out_d(kernel_conf.out_shape(2));
    pooling_3d_ctx->set_out_h(kernel_conf.out_shape(3));
    pooling_3d_ctx->set_out_w(kernel_conf.out_shape(4));

    pooling_3d_ctx->BuildCudnnDescs(this->GetPoolingMode(),
                                    GetDataType<T>::val);
  }
  virtual const Pooling3DKernelConf& GetPooling3DKernelConf() const = 0;
  virtual const PbMessage& GetPooling3DOpConf() const = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_3D_KERNEL_H_
