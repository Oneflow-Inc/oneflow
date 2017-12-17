#include "oneflow/core/operator/multinomial_logistic_loss_op.h"

namespace oneflow {

TEST(MultinomialLogisticLossOp, test_loss_op) {
  OperatorConf op_conf;
  op_conf.set_name("multinomial_logistic_loss_op_test");
  op_conf.mutable_multinomial_logistic_loss_conf()->set_prediction(
      "prediction");
  op_conf.mutable_multinomial_logistic_loss_conf()->set_label("label");
  op_conf.mutable_multinomial_logistic_loss_conf()->set_loss("loss");
  auto loss_op = ConstructOp(op_conf);

  JobConf job_conf;
  job_conf.set_DefaultDataType(DataType::kFloat);
  JobDesc::Singleton()->InitFromJobConf(job_conf);
  HashMap<std::string, BlobDesc*> bn2blob_desc_map = {
      {"prediction", new BlobDesc(Shape({500, 3 * 256 * 256 * 256, 1, 1}),
                                  DataType::kFloat, false)},
      {"label", new BlobDesc(Shape({500, 1, 1, 1}), DataType::kInt32, false)},
      {"loss", new BlobDesc},
      {"loss_buffer", new BlobDesc}};

  auto Bn2BlobDescFunc = [&bn2blob_desc_map](const std::string& bn) {
    return bn2blob_desc_map.at(bn);
  };

  loss_op->InferBlobDescs(Bn2BlobDescFunc, kDataParallel, 2, 10);
  ASSERT_EQ(*Bn2BlobDescFunc("loss"),
            BlobDesc(Shape({1}), DataType::kFloat, false));
  ASSERT_EQ(*Bn2BlobDescFunc("loss_buffer"),
            BlobDesc(Shape({1}), DataType::kFloat, false));
}

}  // namespace oneflow
