#include "oneflow/core/kernel/decode_ofrecord_kernel.h"
#include "oneflow/core/record/ofrecord_decoder.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

void DecodeOFRecordKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  random_seed_gen_.reset(new std::mt19937(kernel_conf().decode_ofrecord_conf().random_seed()));
  distribution_.reset(new std::uniform_int_distribution<int32_t>(0, 1024 * 1024));
  parallel_num_ = parallel_ctx->parallel_num();
}

int32_t DecodeOFRecordKernel::NextRandomInt() const { return (*distribution_)(*random_seed_gen_); }

void DecodeOFRecordKernel::Forward(const KernelCtx& ctx,
                                   std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(ctx.other);
  auto status = static_cast<DecodeStatus*>(ctx.other);
  Blob* in_blob = BnInOp2Blob("in");
  const DecodeOFRecordOpConf& decode_conf = op_conf().decode_ofrecord_conf();
  const size_t out_num = op_attribute().output_bns_size();
  CHECK_EQ(out_num, decode_conf.blob_size());
  status->max_col_id_ = -1;
  FOR_RANGE(int32_t, i, 0, out_num) {
    Blob* out_blob = BnInOp2Blob("out_" + std::to_string(i));
    const BlobConf& blob_conf = decode_conf.blob(i);
    OFRecordDecoderIf* decoder =
        GetOFRecordDecoder(blob_conf.encode_case().encode_case(), blob_conf.data_type());
    int32_t max_col_id =
        decoder->DecodeOneCol(ctx.device_ctx, in_blob, blob_conf, status->cur_col_id_, out_blob,
                              std::bind(&DecodeOFRecordKernel::NextRandomInt, this));

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

REGISTER_KERNEL(OperatorConf::kDecodeOfrecordConf, DecodeOFRecordKernel);

}  // namespace oneflow
