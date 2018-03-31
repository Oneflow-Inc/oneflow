#include "oneflow/core/kernel/decode_random_kernel.h"

namespace oneflow {

int32_t DecodeRandomKernel::GenNextRandomMaxColId() {
  return max_col_id_dis_(max_col_id_gen_);
}

void DecodeRandomKernel::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(ctx.other);
  auto status = static_cast<DecodeStatus*>(ctx.other);
  if(status->max_col_id_ == 0 && status->max_col_id == 0) {
   status->max_col_id_ = GenNextRandomMaxColId(); 
  }
  Blob* out_blob = BnInOp2Blob(kernel_conf().output_bns(0)); 
  const DecodeRandomOpConf& conf = op_conf().decode_random_conf();

  TODO();
  
  
  
  FOR_RANGE(int32_t, i, 0, kernel_conf().output_bns_size()) {
    
    const BlobConf& blob_conf = decode_conf.blob(i);
    OFRecordDecoderIf* decoder =
        GetOFRecordDecoder(blob_conf.encode_case(), blob_conf.data_type());
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

COMMAND(AddKernelCreator(OperatorConf::kDecodeRandomConf,
                         []() { return new DecodeRandomKernel; }));

}  // namespace oneflow
