#include "oneflow/core/rpc_service/common.h"
#include "oneflow/core/kernel/decode_ofrecord_kernel.h"
#include "oneflow/core/record/ofrecord_decoder.h"
#include "oneflow/core/kernel/decode_in_stream_kernel.h"

namespace oneflow {

void DecodeInStreamKernel::Forward(const KernelCtx& ctx,
                                   std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto& predic_params = *static_cast<PredictParams*>(ctx.other);

  auto encode_case = static_cast<EncodeCase>(predic_params.encode_case);
  auto data_type = static_cast<DataType>(predic_params.data_type);

  auto& buffers = predic_params.buffers;
  if (buffers.size() > Global<JobDesc>::Get()->PieceSize()) {
    LOG(ERROR) << "number of predict buffers is more than piecesize";
    return;
  }

  Blob* out_blob = BnInOp2Blob("out");
  out_blob->set_dim0_valid_num(0, buffers.size());

  const auto& conf = op_conf().decode_in_stream_conf().blob(0);

  OFRecordDecoderIf* decoder = GetOFRecordDecoder(encode_case, data_type);
  decoder->Decode(ctx.device_ctx, predic_params, out_blob, conf);
}

REGISTER_KERNEL(OperatorConf::kDecodeInStreamConf, DecodeInStreamKernel);

}  // namespace oneflow