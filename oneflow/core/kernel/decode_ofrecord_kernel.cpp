#include "oneflow/core/kernel/decode_ofrecord_kernel.h"
#include "oneflow/core/record/ofrecord_decoder.h"

namespace oneflow {

void DecodeOFRecordKernel::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(ctx.other);
  auto status = static_cast<DecodeStatus*>(ctx.other);
  auto record_blob = status->in_regst_->GetRecordBlob<OFRecord>();
  const DecodeOFRecordOpConf& decode_conf = op_conf().decode_ofrecord_conf();
  CHECK_EQ(kernel_conf().output_bns_size(), decode_conf.blob_size());
  status->max_col_id_ = -1;
  FOR_RANGE(int32_t, i, 0, kernel_conf().output_bns_size()) {
    Blob* out_blob = BnInOp2Blob(kernel_conf().output_bns(i));
    const BlobConf& blob_conf = decode_conf.blob(i);
    OFRecordDecoderIf* decoder = GetOFRecordDecoder(
        blob_conf.encode_case().encode_case(), blob_conf.data_type());
    int32_t max_col_id = decoder->DecodeOneCol(
        ctx.device_ctx, record_blob, blob_conf, status->cur_col_id_, out_blob);
    if (status->max_col_id_ == -1) {
      status->max_col_id_ = max_col_id;
    } else {
      CHECK_EQ(status->max_col_id_, 0);
      CHECK_EQ(max_col_id, 0);
    }
    CHECK_LT(status->max_col_id_, out_blob->max_col_num());
  }
  CHECK_GE(status->max_col_id_, 0);
}

COMMAND(AddKernelCreator(OperatorConf::kDecodeOfrecordConf,
                         []() { return new DecodeOFRecordKernel; }));

}  // namespace oneflow
