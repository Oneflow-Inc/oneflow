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
    const BlobConf& blob_conf = decode_conf.blob(i);
    const std::string& name = blob_conf.name();
    EncodeType encode_type = blob_conf.encode_type();
    DataType data_type = blob_conf.data_type();
    OFRecordDecoderIf* decoder = GetOFRecordDecoder(encode_type, data_type);
    status->max_col_id_ =
        decoder->Decode(record_blob, name, status->cur_col_id_, out_blob,
                        kernel_ctx.device_ctx);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kDecodeOfrecordConf,
                               DecodeOFRecordKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
