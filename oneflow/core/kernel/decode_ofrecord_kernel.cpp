#include "oneflow/core/kernel/decode_ofrecord_kernel.h"

namespace oneflow {

template<typename T>
void DecodeOFRecordKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(kernel_ctx.other);
  auto status = static_cast<DecodeStatus*>(kernel_ctx.other);

  RecordBlob<OFRecord>* record_blob =
      status->in_regst_->GetRecordBlob<OFRecord>();
  const DecodeOFRecordOpConf& decode_conf = op_conf().decode_ofrecord_conf();
  CHECK_EQ(kernel_conf().output_bns_size(), decode_conf.blob_size());

  FOR_RANGE(int32_t, i, 0, decode_conf.blob_size()) {
    Blob* out_blob = BnInOp2Blob(kernel_conf().output_bns(i));
    const std::string& name = decode_conf.blob(i).name();
    EncodeType encode_type = decode_conf.blob(i).encode_type();
    // TODO
    // generate record_decoder based on the type of encode_type
    // get the datatype of out_blob and pass it to record_decoder
  }
}

}  // namespace oneflow
