#include "oneflow/core/operator/operator.h"

namespace oneflow {

class Dim0DynamicToFixedOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Dim0DynamicToFixedOp);
  Dim0DynamicToFixedOp() = default;
  ~Dim0DynamicToFixedOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_dim0_dynamic_to_fixed_conf());
    int32_t in_size = op_conf().dim0_dynamic_to_fixed_conf().in_size();
    int32_t out_size = op_conf().dim0_dynamic_to_fixed_conf().out_size();
    CHECK_GT(in_size, 0);
    CHECK_EQ(in_size, out_size);
    EnrollRepeatedInputBn("in", in_size, false);
    EnrollRepeatedOutputBn("out", out_size);
    EnrollOutputBn("mask");
  }
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().dim0_dynamic_to_fixed_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    int32_t bn_size = op_conf().dim0_dynamic_to_fixed_conf().in_size();
    int64_t dim0_val = -1;
    FOR_RANGE(int32_t, i, 0, bn_size) {
      const BlobDesc* cur_in = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
      CHECK_OR_RETURN(cur_in->is_dynamic());
      CHECK_EQ_OR_RETURN(0, cur_in->num_of_lod_levels());
      if (dim0_val == -1) {
        dim0_val = cur_in->shape().At(0);
      } else {
        CHECK_EQ_OR_RETURN(dim0_val, cur_in->shape().At(0));
      }

      BlobDesc* cur_out = GetBlobDesc4BnInOp(GenRepeatedBn("out", i));
      *cur_out = *cur_in;
      cur_out->set_is_dynamic(false);
    }
    BlobDesc* mask_blob_desc = GetBlobDesc4BnInOp("mask");
    mask_blob_desc->set_is_dynamic(false);
    mask_blob_desc->set_data_type(DataType::kInt32);
    CHECK_GT_OR_RETURN(dim0_val, 0);
    mask_blob_desc->mut_shape() = Shape(std::vector<int64_t>{dim0_val});
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    FOR_RANGE(int32_t, i, 0, input_bns().size()) {
      OF_CHECK(*BatchAxis4BnInOp(GenRepeatedBn("in", i)) == *BatchAxis4BnInOp(input_bns().Get(0)));
      *BatchAxis4BnInOp(GenRepeatedBn("out", i)) = *BatchAxis4BnInOp(GenRepeatedBn("in", i));
    }
    *BatchAxis4BnInOp("mask") = *BatchAxis4BnInOp(input_bns().Get(0));
    return Maybe<void>::Ok();
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_CPU_OP(OperatorConf::kDim0DynamicToFixedConf, Dim0DynamicToFixedOp);

}  // namespace oneflow
