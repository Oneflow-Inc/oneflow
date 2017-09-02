#include "oneflow/core/operator/softmax_loss_op.h"

namespace oneflow {

TEST(SoftmaxLossOp, softmax_loss_3x5) {
  // create softmax_loss_op
  OperatorConf op_conf;
  op_conf.set_name("softmax_loss_test");
  op_conf.mutable_softmax_loss_conf()->mutable_prediction()->set_name(
      "softmax_loss/prediction");
  op_conf.mutable_softmax_loss_conf()->mutable_prediction()->set_data_type(
      DataType::kFloat);
  op_conf.mutable_softmax_loss_conf()->mutable_label()->set_name(
      "softmax_loss/label");
  op_conf.mutable_softmax_loss_conf()->mutable_label()->set_data_type(
      DataType::kFloat);
  op_conf.mutable_softmax_loss_conf()->mutable_loss()->set_name(
      "softmax_loss/loss");
  op_conf.mutable_softmax_loss_conf()->mutable_loss()->set_data_type(
      DataType::kInt32);
  auto softmax_loss_op = ConstructOp(op_conf);
  JobConf job_conf;
  job_conf.set_default_data_type(DataType::kFloat);
  JobDesc::Singleton()->InitFromJobConf(job_conf);
  HashMap<std::string, BlobDesc*> bn2blob_desc_map{
      {"prediction", new BlobDesc(Shape({3, 5}), DataType::kFloat, false)},
      {"label", new BlobDesc(Shape({3}), DataType::kInt32, false)},
      {"prob", new BlobDesc},
      {"tmp_1D", new BlobDesc},
      {"loss", new BlobDesc}};
  auto fp = [&bn2blob_desc_map](const std::string& bn) {
    return bn2blob_desc_map.at(bn);
  };
  // infershape
  softmax_loss_op->InferBlobDesc4FwBlobs(fp, kDataParallel, 0, 1);
  // test
  ASSERT_EQ(*fp("loss"), BlobDesc(Shape({1}), DataType::kFloat, false));
  ASSERT_EQ(*fp("prob"), BlobDesc(Shape({3, 5}), DataType::kFloat, false));
  ASSERT_EQ(*fp("tmp_1D"), BlobDesc(Shape({3}), DataType::kFloat, false));
}

}  // namespace oneflow
