#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MaskrcnnSplitKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaskrcnnSplitKernel);
  MaskrcnnSplitKernel() = default;
  ~MaskrcnnSplitKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    const int32_t num_segm = this->op_conf().maskrcnn_split_conf().segm_size();
    std::vector<int32_t> dim0_vec;
    FOR_RANGE(int32_t, i, 0, num_segm) {
      dim0_vec.push_back(BnInOp2Blob("segm_" + std::to_string(i))->shape().At(0));
    }
    CHECK_EQ(std::accumulate(dim0_vec.begin(), dim0_vec.end(), 0), in_blob->shape().At(0));
    int32_t offset = 0;
    FOR_RANGE(int32_t, i, 0, num_segm) {
      Memcpy<device_type>(ctx.device_ctx, BnInOp2Blob("out_" + std::to_string(i))->mut_dptr<T>(),
                          in_blob->dptr<T>() + offset,
                          dim0_vec.at(i) * in_blob->shape().Count(1) * sizeof(T));
      offset += dim0_vec.at(i) * in_blob->shape().Count(1);
    }
  }
};

#define REGISTER_MASKRCNN_SPLIT_KERNEL(dtype)                                                      \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kMaskrcnnSplitConf, DeviceType::kCPU, dtype, \
                                        MaskrcnnSplitKernel<DeviceType::kCPU, dtype>)              \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kMaskrcnnSplitConf, DeviceType::kGPU, dtype, \
                                        MaskrcnnSplitKernel<DeviceType::kGPU, dtype>)

REGISTER_MASKRCNN_SPLIT_KERNEL(float);
REGISTER_MASKRCNN_SPLIT_KERNEL(double);
REGISTER_MASKRCNN_SPLIT_KERNEL(int8_t);
REGISTER_MASKRCNN_SPLIT_KERNEL(int32_t);
REGISTER_MASKRCNN_SPLIT_KERNEL(int64_t);

}  // namespace oneflow
