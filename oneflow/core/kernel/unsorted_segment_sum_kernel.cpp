#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/gather_kernel_util.h"
#include "oneflow/core/kernel/unsorted_segment_sum_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
class UnsortedSegmentSumKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnsortedSegmentSumKernel);
  UnsortedSegmentSumKernel() = default;
  ~UnsortedSegmentSumKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T, typename K>
void UnsortedSegmentSumKernel<device_type, T, K>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* segment_ids = BnInOp2Blob("segment_ids");
  const Blob* data = BnInOp2Blob("data");
  Blob* out = BnInOp2Blob("out");
  Memset<device_type>(ctx.device_ctx, out->mut_dptr<T>(), 0, out->ByteSizeOfBlobBody());
  if (segment_ids->IsBodyEmpty() || data->IsBodyEmpty()) { return; }
  const ShapeView& out_shape = out->shape();
  const int64_t axis = this->kernel_conf().unsorted_segment_sum_conf().axis();
  const int64_t outer_dim_size = out_shape.Count(0, axis);
  const int64_t num_segments = out_shape.At(axis);
  const int64_t inner_dim_size = out_shape.Count(axis + 1);
  const int64_t num_segment_ids = segment_ids->shape().elem_cnt();
  CHECK_EQ(inner_dim_size * num_segment_ids * outer_dim_size, data->shape().elem_cnt());
  UnsortedSegmentSumKernelUtil<device_type, T, K>::UnsortedSegmentSum(
      ctx.device_ctx, segment_ids->dptr<K>(), data->dptr<T>(), num_segment_ids, num_segments,
      outer_dim_size, inner_dim_size, 0, out->mut_dptr<T>());
}

namespace {

#define MAKE_UNSORTED_SEGMENT_SUM_KERNEL_ENTRY(op_type_case, device_type_v, data_type_pair,     \
                                               indices_type_pair)                               \
  NEW_REGISTER_KERNEL(op_type_case,                                                             \
                      UnsortedSegmentSumKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair), \
                                               OF_PP_PAIR_FIRST(indices_type_pair)>)            \
      .SetIsMatchedPred([](const KernelConf& kernel_conf) -> bool {                             \
        return ((kernel_conf.op_attribute().op_conf().device_type() == device_type_v)           \
                && ((OF_PP_PAIR_SECOND(data_type_pair)) == kernel_conf.data_type())             \
                && (OF_PP_PAIR_SECOND(indices_type_pair)                                        \
                    == kernel_conf.unsorted_segment_sum_conf().indices_data_type()));           \
      });
#define MAKE_UNSORTED_SEGMENT_SUM_OP_TYPE_CASE_SEQ            \
  OF_PP_MAKE_TUPLE_SEQ(OperatorConf::kUnsortedSegmentSumConf) \
  OF_PP_MAKE_TUPLE_SEQ(OperatorConf::kUnsortedSegmentSumLikeConf)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_UNSORTED_SEGMENT_SUM_KERNEL_ENTRY,
                                 MAKE_UNSORTED_SEGMENT_SUM_OP_TYPE_CASE_SEQ, DEVICE_TYPE_SEQ,
                                 UNSORTED_SEGMENT_SUM_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#undef MAKE_UNSORTED_SEGMENT_SUM_OP_TYPE_CASE_SEQ
#undef MAKE_UNSORTED_SEGMENT_SUM_KERNEL_ENTRY

}  // namespace

}  // namespace oneflow
