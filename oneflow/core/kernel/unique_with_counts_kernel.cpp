#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/unique_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
class UniqueWithCountsKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UniqueWithCountsKernel);
  UniqueWithCountsKernel() = default;
  ~UniqueWithCountsKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T, typename K>
void UniqueWithCountsKernel<device_type, T, K>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

#define MAKE_UNIQUE_WITH_COUNTS_KERNEL_ENTRY(device_type_v, data_type_pair, indices_type_pair) \
  NEW_REGISTER_KERNEL(OperatorConf::kUniqueWithCountsConf,                                     \
                      UniqueWithCountsKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair),  \
                                             OF_PP_PAIR_FIRST(indices_type_pair)>)             \
      .SetIsMatchedPred([](const KernelConf& kernel_conf) -> bool {                            \
        return ((kernel_conf.op_attribute().op_conf().device_type() == device_type_v)          \
                && ((OF_PP_PAIR_SECOND(data_type_pair)) == kernel_conf.data_type())            \
                && (OF_PP_PAIR_SECOND(indices_type_pair)                                       \
                    == kernel_conf.unique_with_counts_conf().indices_data_type()));            \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_UNIQUE_WITH_COUNTS_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#undef MAKE_UNIQUE_WITH_COUNTS_KERNEL_ENTRY

}  // namespace oneflow
