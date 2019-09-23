#include "oneflow/core/operator/operator.h"

namespace oneflow {

class DynamicReshapeOp final : public Operator {
 public:
  void InitFromOpConf() {
    CHECK(op_conf().has_dynamic_reshape_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out")->set_const_inplace_ibn("in");
  }
  const PbMessage& GetCustomizedConf() const { return op_conf().dynamic_reshape_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx,
                             const SbpSignature* sbp_signature) const {
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    *out = *in;
    const DynamicReshapeOpConf& conf = op_conf().dynamic_reshape_conf();
    int32_t inferred_axis = -1;
    int32_t num_inferred_axis = 0;
    std::vector<int64_t> conf_dim_vec = {conf.shape().dim().begin(), conf.shape().dim().end()};
    std::vector<int64_t> out_dim_vec = {};
    for (int32_t i = 0; i < dim_vec.size(); ++i) {
      if (conf_dim_vec[i] == -1) {
        inferred_axis = i;
        num_inferred_axis = 1;
      } else {
        CHECK_GT_OR_RETURN(conf_dim_vec[i], 0);
        out_dim_vec.push_back(conf_dim_vec[i]);
      }
    }
    CHECK_GT_OR_RETURN(inferred_axis, 0);
    CHECK_EQ_OR_RETURN(num_inferred_axis, 1);
    out_dim_vec.insert(out_dim_vec.begin() + inferred_axis,
                   in->shape().elem_cnt() / out->shape().elem_cnt());
    out->mut_shape() = Shape(out_dim_vec);
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
};

REGISTER_OP(OperatorConf::kDynamicReshapeConf, DynamicReshapeOp);

}  // namespace oneflow
