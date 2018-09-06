#include "oneflow/core/kernel/reduce_local_add2_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceLocalAdd2Kernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto* other_val = static_cast<std::tuple<int64_t, bool, bool, bool>*>(ctx.other);
  int32_t in_bn_id = std::get<0>(*other_val);
  bool is_inited = std::get<1>(*other_val);
  bool is_inplace_in_blob = std::get<2>(*other_val);
  bool enable_inplace = std::get<3>(*other_val);

  if (is_inplace_in_blob) { return; }

  if (enable_inplace) {
    Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(in_bn_id));
    Blob* first_out_blob = BnInOp2Blob(this->op_attribute().output_bns().Get(0));
    if (is_inited) {
      KernelUtil<device_type, T>::Axpy(ctx.device_ctx, in_blob->shape().elem_cnt(), 1.0,
                                       in_blob->dptr<T>(), 1, first_out_blob->mut_dptr<T>(), 1);
    } else {
      Memcpy<device_type>(ctx.device_ctx, first_out_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                          in_blob->ByteSizeOfDataContentField());
    }
  } else {
    Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(in_bn_id));
    if (is_inited) {
      int64_t offset = 0;
      FOR_RANGE(int32_t, i, 0, this->op_attribute().output_bns().size()) {
        Blob* out_blob = BnInOp2Blob(this->op_attribute().output_bns().Get(i));
        KernelUtil<device_type, T>::Axpy(ctx.device_ctx, out_blob->shape().elem_cnt(), 1.0,
                                         in_blob->dptr<T>() + offset, 1, out_blob->mut_dptr<T>(),
                                         1);
        offset += out_blob->shape().elem_cnt();
      }
    } else {
      int64_t offset = 0;
      FOR_RANGE(int32_t, i, 0, this->op_attribute().output_bns().size()) {
        Blob* out_blob = BnInOp2Blob(this->op_attribute().output_bns().Get(i));
        Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(),
                            in_blob->dptr<char>() + offset, out_blob->ByteSizeOfDataContentField());
        offset += out_blob->ByteSizeOfDataContentField();
      }
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceLocalAdd2Conf, ReduceLocalAdd2Kernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
