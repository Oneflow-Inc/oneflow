#include "oneflow/core/kernel/rnn_lookup_kernel.h"

namespace oneflow {

template<typename IntegerType, typename FloatType>
void RnnLookupKernel<IntegerType, FloatType>::Forward(const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_blob = BnInOp2Blob("out");
  const Blob* weight_blob = BnInOp2Blob("weight");

  int64_t piece_size = in_blob->shape().At(0);
  CHECK_EQ(piece_size, out_blob->shape().At(0));
  int64_t hidden_dim = out_blob->shape().At(1);
  CHECK_EQ(hidden_dim, weight_blob->shape().At(1));
  int64_t vocab_size = weight_blob->shape().At(0);

  CopyDataIdFromSoleIbToAllObIfNeed<DeviceType::kCPU>(ctx, BnInOp2Blob);

  const IntegerType* in_dptr = in_blob->dptr<IntegerType>();
  const FloatType* weight_dptr = weight_blob->dptr<FloatType>();
  FloatType* out_start_dptr = out_blob->mut_dptr<FloatType>();

  for (int64_t i = 0;i < piece_size; ++i) {
    FloatType* out_dptr = out_start_dptr + i * hidden_dim;

    IntegerType vocab_id = *in_dptr;
    //TODO: no need to send to cpu stream?
    if (vocab_id == -1) {
      memset(out_dptr, 0, sizeof(FloatType) * hidden_dim);
    } else {
      memcpy(out_dptr, weight_dptr + vocab_id * hidden_dim, hidden_dim * sizeof(FloatType));
    }
  }
}

template<typename IntegerType, typename FloatType>
void RnnLookupKernel<IntegerType, FloatType>::Backward(const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");

  int64_t piece_size = in_blob->shape().At(0);
  CHECK_EQ(piece_size, out_diff_blob->shape().At(0));
  int64_t hidden_dim = out_diff_blob->shape().At(1);
  CHECK_EQ(hidden_dim, weight_diff_blob->shape().At(1));
  int64_t vocab_size = weight_diff_blob->shape().At(0);

  const IntegerType* in_dptr = in_blob->dptr<IntegerType>();
  const FloatType* out_diff_dptr = out_diff_blob->dptr<FloatType>();
  FloatType* weight_diff_start_dptr = weight_diff_blob->mut_dptr<FloatTyp>();
  memset(weight_diff_start_blob, 0, sizeof(FloatType) * hidden_dim * vocab_size);

  for (int64_t i = 0;i < piece_size; ++i) {
    IntegerType vocab_id = *in_dptr;
    FloatType* weight_diff_dptr = weight_diff_start_dptr + vocab_id * hidden_dim;
    if (vocab_id == -1) { continue; }
    KernelUtil<DeviceType::kCPU, FloatType>::BlasAxpy(
        ctx.device_ctx, hidden_dim, 1.0, out_dptr + i * hidden_dim, 1, weight_diff_dptr, 1);
  }

}

Kernel* CreateRnnLookupKernel(const OpContext& op_ctx) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define RNN_LOOKUP_KERNEL_ENTRY(integer_data_type_pair, float_data_type_pair) \
    {GetHashKey( \
       OF_PP_PAIR_SECOND(integer_data_type_pair), \ 
       OF_PP_PAIR_SECOND(float_data_type_pair) \
     ), \
     []() { return new RnnLookupKernel< \
       OF_PP_PAIR_FIRST(integer_data_type_pair), \
       OF_PP_PAIR_FIRST(float_data_type_pair)>; \
     }},
     OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
         RNN_LOOKUP_KERNEL_ENTRY, SIGNED_INT_DATA_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)};
  return creators.at(
      GetHashKey(op_ctx.bn_in_op2data_type().at("in"), 
                 op_ctx.bn_in_op2data_type().at("out")))();
}

COMMAND(AddKernelCreator(OperatorConf::kRnnLookupConf,
                        CreateRnnLookupKernel));

} // namespace oneflow
