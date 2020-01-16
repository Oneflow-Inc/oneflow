#include "oneflow/core/kernel/unsorted_segment_sum_kernel.h"
#include "oneflow/core/kernel/gather_kernel_util.h"
#include "oneflow/core/kernel/unsorted_segment_sum_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
const PbMessage& UnsortedSegmentSumKernel<device_type, T, K>::GetCustomizedOpConf() const {
  return this->op_conf().unsorted_segment_sum_conf();
}

template<DeviceType device_type, typename T, typename K>
void UnsortedSegmentSumKernel<device_type, T, K>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* segment_ids = BnInOp2Blob("segment_ids");
  const Blob* data = BnInOp2Blob("data");
  Blob* out = BnInOp2Blob("out");
  Memset<device_type>(ctx.device_ctx, out->mut_dptr<T>(), 0, out->ByteSizeOfBlobBody());
  if (segment_ids->IsBodyEmpty() || data->IsBodyEmpty()) { return; }
  const ShapeView& out_shape = out->shape();
  const int64_t axis = this->op_conf().unsorted_segment_sum_conf().axis();
  const int64_t outer_dim_size = out_shape.Count(0, axis);
  const int64_t num_segments = this->op_conf().unsorted_segment_sum_conf().num_segments();
  CHECK_EQ(out_shape.At(axis), num_segments);
  const int64_t inner_dim_size = out_shape.Count(axis + 1);
  const int64_t num_segment_ids = segment_ids->shape().elem_cnt();
  CHECK_EQ(inner_dim_size * num_segment_ids * outer_dim_size, data->shape().elem_cnt());
  UnsortedSegmentSumKernelUtil<device_type, T, K>::UnsortedSegmentSum(
      ctx.device_ctx, segment_ids->dptr<K>(), data->dptr<T>(), num_segment_ids, num_segments,
      outer_dim_size, inner_dim_size, 0, out->mut_dptr<T>());

}  // namespace oneflow

namespace {

#define MAKE_UNSORTED_SEGMENT_SUM_KERNEL_ENTRY(device_type_v, data_type_pair, indices_type_pair) \
  NEW_REGISTER_KERNEL(OperatorConf::kUnsortedSegmentSumConf,                                     \
                      UnsortedSegmentSumKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair),  \
                                               OF_PP_PAIR_FIRST(indices_type_pair)>)             \
      .SetIsMatchedPred([](const KernelConf& kernel_conf) -> bool {                              \
        return ((kernel_conf.op_attribute().op_conf().device_type() == device_type_v)            \
                && ((OF_PP_PAIR_SECOND(data_type_pair)) == kernel_conf.data_type())              \
                && (OF_PP_PAIR_SECOND(indices_type_pair)                                         \
                    == kernel_conf.unsorted_segment_sum_conf().indices_data_type()));            \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_UNSORTED_SEGMENT_SUM_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                 UNSORTED_SEGMENT_SUM_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#undef MAKE_UNSORTED_SEGMENT_SUM_KERNEL_ENTRY

}  // namespace

}  // namespace oneflow
