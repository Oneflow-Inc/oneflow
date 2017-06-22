#include "oneflow/core/operator/multinomial_logistic_loss_op.h"

namespace oneflow {


TEST(MultinomialLogisticLossOp, test_loss_op) {
  OperatorConf op_conf;
  op_conf.set_name("multinomial_logistic_loss_op_test");
  op_conf.mutable_multinomial_logistic_loss_conf()
    ->set_prediction("prediction");
  op_conf.mutable_multinomial_logistic_loss_conf()->set_label("label");
  op_conf.mutable_multinomial_logistic_loss_conf()->set_loss("loss");
  auto loss_op = OpMgr::Singleton().ConstructOp(op_conf);

  HashMap<std::string, Shape*> bn2shape_ptr{
    {loss_op->input_bns().at(0), new Shape({500, 3*256*256*256, 1, 1})},
    {loss_op->input_bns().at(1), new Shape({500, 1, 1, 1})},
    {loss_op->input_diff_bns().at(0), new Shape},
    {loss_op->input_diff_bns().at(1), new Shape},
    {loss_op->SoleObn(), new Shape},
    {loss_op->SoleDtbn(), new Shape}};

  auto fp = [&bn2shape_ptr](const std::string& bn) {
    return bn2shape_ptr.at(bn);
  };

  loss_op-> InferShape4FwBlobs(fp, kDataParallel, 2, 10);

  Shape* loss_shape_ptr = bn2shape_ptr.at(loss_op->SoleObn());
  Shape* loss_buffer_shape_ptr = bn2shape_ptr.at(loss_op->SoleDtbn());
  ASSERT_EQ(*loss_shape_ptr, Shape({1}));
  ASSERT_EQ(*loss_buffer_shape_ptr, Shape({1}));
  Shape* prediction_diff_shape_ptr = 
    bn2shape_ptr.at(loss_op->input_diff_bns().at(0));
  ASSERT_EQ(*prediction_diff_shape_ptr, 
            *bn2shape_ptr.at(loss_op->input_bns().at(0)));
  Shape* label_diff_shape_ptr = 
    bn2shape_ptr.at(loss_op->input_diff_bns().at(1));
  ASSERT_EQ(*label_diff_shape_ptr, 
            *bn2shape_ptr.at(loss_op->input_bns().at(1)));
}

}  // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
