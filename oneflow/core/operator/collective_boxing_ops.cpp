#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"

namespace oneflow {

using namespace boxing::collective;

class CollectiveBoxingGenericOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingGenericOp);
  CollectiveBoxingGenericOp() = default;
  ~CollectiveBoxingGenericOp() override = default;

 private:
  void InitFromOpConf() override {
    CHECK(op_conf().has_collective_boxing_generic_conf());
    const RankDesc& rank_desc = op_conf().collective_boxing_generic_conf().rank_desc();
    if (GenericOpHasInput(rank_desc)) { EnrollInputBn("in", false); }
    if (GenericOpHasOutput(rank_desc)) { EnrollOutputBn("out", false); }
  }

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().collective_boxing_generic_conf();
  }

  LogicalBlobId lbi4ibn(const std::string& input_bn) const override {
    return this->op_conf().collective_boxing_generic_conf().lbi();
  }

  LogicalBlobId lbi4obn(const std::string& output_bn) const override {
    return this->op_conf().collective_boxing_generic_conf().lbi();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext*) const override {
    const RankDesc& rank_desc = op_conf().collective_boxing_generic_conf().rank_desc();
    const DataType data_type = rank_desc.op_desc().data_type();
    if (GenericOpHasInput(rank_desc)) {
      const BlobDesc* in = GetBlobDesc4BnInOp("in");
      CHECK_OR_RETURN(!in->is_dynamic());
      CHECK_EQ_OR_RETURN(in->data_type(), data_type);
      CHECK_EQ_OR_RETURN(in->shape(), GenericOpGetInputShape(rank_desc));
    }
    if (GenericOpHasOutput(rank_desc)) {
      BlobDesc* out = GetBlobDesc4BnInOp("out");
      out->set_data_type(data_type);
      out->mut_shape() = GenericOpGetOutputShape(rank_desc);
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kCollectiveBoxingGenericConf, CollectiveBoxingGenericOp);

}  // namespace oneflow
