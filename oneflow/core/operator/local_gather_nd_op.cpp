#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LocalGatherNdOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalGatherNdOp);
  LocalGatherNdOp() = default;
  ~LocalGatherNdOp() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_local_gather_nd_conf());
    EnrollInputBn("in");
    EnrollInputBn("indices", false);
    if (this->device_type() == DeviceType::kGPU) { EnrollTmpBn("shape"); }
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().local_gather_nd_conf(); }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    GatherNdUpdateKernelConf* conf = kernel_conf->mutable_gather_nd_update_conf();
    conf->set_idx_type(GetBlobDesc4BnInOp("indices")->data_type());
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    // input: in, indices, updates
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    const BlobDesc* indices = GetBlobDesc4BnInOp("indices");

    OF_CHECK(IsIntegralDataType(indices->data_type()));

    int64_t segm_dims = indices->shape().At(indices->shape().NumAxes() - 1);
    OF_CHECK_LE(segm_dims, in->shape().NumAxes());
    DimVector shape_vec = indices->shape().dim_vec();
    shape_vec.pop_back();
    FOR_RANGE(int64_t, i, segm_dims, in->shape().NumAxes()) {
      shape_vec.push_back(in->shape().At(i));
    }

    BlobDesc* shape = GetBlobDesc4BnInOp("shape");
    if (shape) {
      shape->mut_shape() = Shape({in->shape().NumAxes()});
      shape->set_data_type(DataType::kInt64);
    }

    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->set_is_dynamic(in->is_dynamic() || indices->is_dynamic());
    out->mut_shape() = Shape(shape_vec);
    out->set_data_type(in->data_type());
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kLocalGatherNdConf, LocalGatherNdOp);

}  // namespace oneflow
