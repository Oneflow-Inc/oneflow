#ifndef ONEFLOW_CORE_KERNEL_POOLING_2D_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_2D_KERNEL_H_

#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class Pooling2DKernel : public PoolingKernel<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling2DKernel);
  Pooling2DKernel() = default;
  virtual ~Pooling2DKernel() = default;

 protected:
  void VirtualKernelInit(const ParallelContext*) override {
    Pooling3DCtx* pooling_3d_ctx = this->mut_pooling_3d_ctx();
    const PbMessage& op_conf = GetPooling2DOpConf();
    const PbRf<int32_t>& pool_size =
        GetPbRfFromPbMessage<int32_t>(op_conf, "pool_size");
    pooling_3d_ctx->set_pool_size_d(1);
    pooling_3d_ctx->set_pool_size_h(pool_size.Get(0));
    pooling_3d_ctx->set_pool_size_w(pool_size.Get(1));
    const PbRf<int32_t>& strides =
        GetPbRfFromPbMessage<int32_t>(op_conf, "strides");
    pooling_3d_ctx->set_strides_d(1);
    pooling_3d_ctx->set_strides_h(strides.Get(0));
    pooling_3d_ctx->set_strides_w(strides.Get(1));

    const Pooling2DKernelConf& kernel_conf = GetPooling2DKernelConf();
    pooling_3d_ctx->set_padding_d(0);
    pooling_3d_ctx->set_padding_h(kernel_conf.padding_h());
    pooling_3d_ctx->set_padding_w(kernel_conf.padding_w());

    pooling_3d_ctx->set_in_n(kernel_conf.in_shape(0));
    pooling_3d_ctx->set_in_c(kernel_conf.in_shape(1));
    pooling_3d_ctx->set_in_d(1);
    pooling_3d_ctx->set_in_h(kernel_conf.in_shape(2));
    pooling_3d_ctx->set_in_w(kernel_conf.in_shape(3));

    pooling_3d_ctx->set_out_n(kernel_conf.out_shape(0));
    pooling_3d_ctx->set_out_c(kernel_conf.out_shape(1));
    pooling_3d_ctx->set_out_d(1);
    pooling_3d_ctx->set_out_h(kernel_conf.out_shape(2));
    pooling_3d_ctx->set_out_w(kernel_conf.out_shape(3));

    pooling_3d_ctx->BuildCudnnDescs(this->GetPoolingMode(),
                                    GetDataType<T>::val);
  }
  virtual const Pooling2DKernelConf& GetPooling2DKernelConf() const = 0;
  virtual const PbMessage& GetPooling2DOpConf() const = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_2D_KERNEL_H_
