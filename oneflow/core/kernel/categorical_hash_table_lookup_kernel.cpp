#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/categorical_hash_table_lookup_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename K, typename V>
class CategoricalHashTableLookupKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CategoricalHashTableLookupKernel);
  CategoricalHashTableLookupKernel() = default;
  ~CategoricalHashTableLookupKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename K, typename V>
void CategoricalHashTableLookupKernel<device_type, K, V>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(this->op_conf().categorical_hash_table_lookup_conf().hash_precomputed());
  Blob* key = BnInOp2Blob("key");
  Blob* value = BnInOp2Blob("value");
  Blob* size = BnInOp2Blob("size");
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  CategoricalHashTableLookupKernelUtil<device_type, K, V>::GetOrInsert(
      ctx.device_ctx, key->shape().elem_cnt(), key->mut_dptr<K>(), value->mut_dptr<V>(),
      size->mut_dptr<V>(), in->shape().elem_cnt(), in->dptr<K>(), out->mut_dptr<V>());
}

namespace {

#define MAKE_CATEGORICAL_HASH_TABLE_LOOKUP_KERNEL_ENTRY(device_type_v, key_type_pair,      \
                                                        value_type_pair)                   \
  NEW_REGISTER_KERNEL(                                                                     \
      OperatorConf::kCategoricalHashTableLookupConf,                                       \
      CategoricalHashTableLookupKernel<device_type_v, OF_PP_PAIR_FIRST(key_type_pair),     \
                                       OF_PP_PAIR_FIRST(value_type_pair)>)                 \
      .SetIsMatchedPred([](const KernelConf& kernel_conf) -> bool {                        \
        return ((kernel_conf.op_attribute().op_conf().device_type() == device_type_v)      \
                && ((OF_PP_PAIR_SECOND(value_type_pair)) == kernel_conf.data_type())       \
                && (OF_PP_PAIR_SECOND(key_type_pair)                                       \
                    == kernel_conf.categorical_hash_table_lookup_conf().key_data_type())); \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_CATEGORICAL_HASH_TABLE_LOOKUP_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#undef MAKE_CATEGORICAL_HASH_TABLE_LOOKUP_KERNEL_ENTRY

}  // namespace

}  // namespace oneflow
