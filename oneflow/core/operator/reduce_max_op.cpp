#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ReduceMaxOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceMaxOp);
  ReduceMaxOp() = default;
  ~ReduceMaxOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_reduce_max_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out");
    EnrollTmpBn("fw_tmp");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().reduce_max_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const ReduceMaxOpConf& conf = op_conf().reduce_max_conf();
    const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
    *GetBlobDesc4BnInOp("fw_tmp") = *in_blob;
    BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
    out_blob->set_data_type(in_blob->data_type());
    if (conf.axis().empty()) {
      if (conf.keep_dims()) {
        out_blob->mut_shape() = Shape::Ones(in_blob->shape().NumAxes());
      } else {
        out_blob->mut_shape() = Shape({1});
      }
    } else {
      const std::vector<int64_t> axis_vec = {conf.axis().begin(), conf.axis().end()};
      const Shape& reduced_shape = in_blob->shape().CreateReducedShape(axis_vec);
      if (conf.keep_dims()) {
        out_blob->mut_shape() = reduced_shape;
      } else {
        out_blob->mut_shape() = reduced_shape.RemoveOnes(axis_vec);
      }
    }
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    const auto& reduced_axes = op_conf().reduce_max_conf().axis();
    HashSet<int64_t> conf_axes = {reduced_axes.begin(), reduced_axes.end()};
    if (BatchAxis4BnInOp("in")->has_value() && !conf_axes.empty()
        && conf_axes.find(BatchAxis4BnInOp("in")->value()) == conf_axes.end()) {
      *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
    } else {
      BatchAxis4BnInOp("out")->clear_value();
    }
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

class DeviceReduceMaxOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceReduceMaxOp);
  DeviceReduceMaxOp() = default;
  ~DeviceReduceMaxOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_device_reduce_max_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out");
    EnrollTmpBn("fw_tmp");
  }
  const PbMessage& GetCustomizedConf() const override { return op_conf().device_reduce_max_conf(); }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const DeviceReduceMaxOpConf& conf = op_conf().device_reduce_max_conf();
    const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
    *GetBlobDesc4BnInOp("fw_tmp") = *in_blob;
    BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
    out_blob->set_data_type(in_blob->data_type());
    if (conf.axis().empty()) {
      out_blob->mut_shape() = Shape::Ones(in_blob->shape().NumAxes());
    } else {
      const std::vector<int64_t> axis_vec = {conf.axis().begin(), conf.axis().end()};
      const Shape& reduced_shape = in_blob->shape().CreateReducedShape(axis_vec);
      out_blob->mut_shape() = reduced_shape;
    }
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    const auto& reduced_axes = op_conf().device_reduce_max_conf().axis();
    HashSet<int64_t> conf_axes = {reduced_axes.begin(), reduced_axes.end()};
    if (BatchAxis4BnInOp("in")->has_value() && !conf_axes.empty()
        && conf_axes.find(BatchAxis4BnInOp("in")->value()) == conf_axes.end()) {
      *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
    } else {
      BatchAxis4BnInOp("out")->clear_value();
    }
    return Maybe<void>::Ok();
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split(input_bns(), 1)
        .Split(output_bns(), 1)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kReduceMaxConf, ReduceMaxOp);
REGISTER_OP(OperatorConf::kDeviceReduceMaxConf, DeviceReduceMaxOp);

}  // namespace oneflow
