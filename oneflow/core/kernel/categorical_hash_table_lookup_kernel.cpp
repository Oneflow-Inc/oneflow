#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/categorical_hash_table_lookup_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class CategoricalHashTableLookupKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CategoricalHashTableLookupKernel);
  CategoricalHashTableLookupKernel() = default;
  ~CategoricalHashTableLookupKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
void CategoricalHashTableLookupKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(this->op_conf().categorical_hash_table_lookup_conf().hash_precomputed());
  Blob* table = BnInOp2Blob("table");
  Blob* size = BnInOp2Blob("size");
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const int64_t table_elem_cnt = table->shape().elem_cnt();
  CHECK_EQ(table_elem_cnt % 2, 0);
  const int64_t capacity = table_elem_cnt / 2;
  CategoricalHashTableLookupKernelUtil<device_type, T>::GetOrInsert(
      ctx.device_ctx, capacity, table->mut_dptr<T>(), size->mut_dptr<T>(), in->shape().elem_cnt(),
      in->dptr<T>(), out->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kCategoricalHashTableLookupConf,
                           CategoricalHashTableLookupKernel, INDEX_DATA_TYPE_SEQ);

}  // namespace oneflow
