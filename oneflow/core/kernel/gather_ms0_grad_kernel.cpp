#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/unsorted_segment_sum_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
class GatherMs0GradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherMs0GradKernel);
  GatherMs0GradKernel() = default;
  ~GatherMs0GradKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T, typename K>
const PbMessage& GatherMs0GradKernel<device_type, T, K>::GetCustomizedOpConf() const {
  return this->op_conf().gather_ms0_grad_conf();
}

template<DeviceType device_type, typename T, typename K>
void GatherMs0GradKernel<device_type, T, K>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices = BnInOp2Blob("indices");
  const Blob* out_diff = BnInOp2Blob("out_diff");
  Blob* in_diff = BnInOp2Blob("in_diff");
  const int64_t offset = this->kernel_conf().gather_ms0_grad_conf().offset();
  Memset<device_type>(ctx.device_ctx, in_diff->mut_dptr<T>(), 0, in_diff->ByteSizeOfBlobBody());
  const int64_t num_segment_ids = indices->shape().elem_cnt();
  const ShapeView& in_diff_shape = in_diff->shape();
  const int64_t inner_dim_size = in_diff_shape.Count(1);
  const int64_t num_segments = in_diff_shape.At(0);
  CHECK_EQ(out_diff->shape().elem_cnt(), num_segment_ids * inner_dim_size);
  UnsortedSegmentSumKernelUtil<device_type, T, K>::UnsortedSegmentSum(
      ctx.device_ctx, indices->dptr<K>(), out_diff->dptr<T>(), num_segment_ids, num_segments, 1,
      inner_dim_size, offset, in_diff->mut_dptr<T>());
}

namespace {

#define MAKE_GATHER_MS0_GRAD_KERNEL_ENTRY(device_type_v, data_type_pair, indices_type_pair) \
  NEW_REGISTER_KERNEL(OperatorConf::kGatherMs0GradConf,                                     \
                      GatherMs0GradKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair),  \
                                          OF_PP_PAIR_FIRST(indices_type_pair)>)             \
      .SetIsMatchedPred([](const KernelConf& kernel_conf) -> bool {                         \
        return ((kernel_conf.op_attribute().op_conf().device_type() == device_type_v)       \
                && ((OF_PP_PAIR_SECOND(data_type_pair)) == kernel_conf.data_type())         \
                && (OF_PP_PAIR_SECOND(indices_type_pair)                                    \
                    == kernel_conf.gather_ms0_grad_conf().indices_data_type()));            \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_GATHER_MS0_GRAD_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                 UNSORTED_SEGMENT_SUM_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#undef MAKE_GATHER_MS0_GRAD_KERNEL_ENTRY

}  // namespace

}  // namespace oneflow
