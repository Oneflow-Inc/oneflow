#include "oneflow/core/operator/operator.h"
#include "oneflow/core/kernel/argwhere_util.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

Maybe<size_t> InferArgwhereTmpBufferSize(const BlobDesc* in_desc, DataType out_data_type) {
  int64_t elem_cnt = in_desc->shape().elem_cnt();
  size_t tmp_bytes = 0;

#define MAKE_INFER_ARGWHERE_TMP_FN_PAIR_ENTRY(dtype_pair, itype_pair)               \
  {GetHashKey(OF_PP_PAIR_SECOND(dtype_pair), OF_PP_PAIR_SECOND(itype_pair)),        \
   [](int elem_cnt, size_t& tmp_bytes) {                                            \
     CudaCheck(InferCubSelectFlaggedTempStorageBytes<OF_PP_PAIR_FIRST(dtype_pair),  \
                                                     OF_PP_PAIR_FIRST(itype_pair)>( \
         nullptr, elem_cnt, tmp_bytes));                                            \
   }},

  static const HashMap<std::string, std::function<void(int, size_t&)>> infer_fn_map = {
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_INFER_ARGWHERE_TMP_FN_PAIR_ENTRY,
                                       ARGWHERE_SUPPORTED_DATA_TYPE_SEQ,
                                       ARGWHERE_SUPPORTED_INDEX_TYPE_SEQ)};
#undef MAKE_INFER_ARGWHERE_TMP_FN_PAIR_ENTRY

  auto infer_fn_it = infer_fn_map.find(GetHashKey(in_desc->data_type(), out_data_type));
  OF_CHECK(infer_fn_it != infer_fn_map.end())
      << "Argwhere op do not support data_type (" << in_desc->data_type() << "), index_type ("
      << out_data_type << ")";

  infer_fn_it->second(elem_cnt, tmp_bytes);
  return tmp_bytes;
}

}  // namespace

class ArgwhereOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArgwhereOp);
  ArgwhereOp() = default;
  ~ArgwhereOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_argwhere_conf());
    EnrollInputBn("in", false);
    EnrollOutputBn("out", false);
    EnrollOutputBn("out_size", false);
    EnrollTmpBn("tmp");
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().argwhere_conf(); }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature,
                                std::function<void(OpContext*)> EnrollOpCtx) const override {
    const BlobDesc* in_desc = GetBlobDesc4BnInOp("in");
    const int64_t elem_cnt = in_desc->shape().elem_cnt();
    const DataType out_data_type = op_conf().argwhere_conf().data_type();
    OF_CHECK(IsIntegralDataType(out_data_type));
    BlobDesc* out_desc = GetBlobDesc4BnInOp("out");
    out_desc->mut_shape() = Shape({elem_cnt, in_desc->shape().NumAxes()});
    out_desc->set_data_type(out_data_type);
    BlobDesc* out_size_desc = GetBlobDesc4BnInOp("out_size");
    out_size_desc->mut_shape() = Shape({1});
    out_size_desc->set_data_type(DataType::kInt32);
    return Maybe<void>::Ok();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    InferOutBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, sbp_signature, EnrollOpCtx);
    BlobDesc* tmp_desc = GetBlobDesc4BnInOp("tmp");
    size_t tmp_bytes = JUST(InferArgwhereTmpBufferSize(GetBlobDesc4BnInOp("in"),
                                                       op_conf().argwhere_conf().data_type()));
    OF_CHECK_GT(tmp_bytes, 0) << "tmp buffer bytes should be greater than zero";
    tmp_desc->mut_shape() = Shape({tmp_bytes});
    tmp_desc->set_data_type(DataType::kChar);
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf* kernel_conf) const override {
    const BlobDesc* in_desc = GetBlobDesc4BnInOp("in");
    const BlobDesc* out_desc = GetBlobDesc4BnInOp("out");
    kernel_conf->set_data_type(out_desc->data_type());
    kernel_conf->mutable_argwhere_gpu_conf()->set_num_axes(in_desc->shape().NumAxes());
    kernel_conf->mutable_argwhere_gpu_conf()->set_data_type(in_desc->data_type());
    kernel_conf->mutable_argwhere_gpu_conf()->set_index_type(out_desc->data_type());
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    for (const std::string& obn : output_bns()) { BatchAxis4BnInOp(obn)->set_value(0); }
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("in", 0)
        .Split("out", 0)
        .Broadcast("out_size")
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kArgwhereConf, ArgwhereOp);

}  // namespace oneflow
