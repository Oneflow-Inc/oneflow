#include "operator/multinomial_logistic_loss_op.h"
#include "operator/operator_manager.h"
#include "gtest/gtest.h"

namespace oneflow {


TEST(MultinomialLogisticLossOp, test_loss_op) {
  /*
  OperatorConf op_conf;
  op_conf.set_name("multinomial_logistic_loss_op_test");
  op_conf.mutable_multinomial_logistic_loss_conf()->set_data("data");
  op_conf.mutable_multinomial_logistic_loss_conf()->set_label("label");
  op_conf.mutable_multinomial_logistic_loss_conf()->set_loss("loss");
  auto loss_op = OpMgr::Singleton().ConstructOp(op_conf);

  TestShapeFactory shape_manager = TestShapeFactory();
  std::vector<int64_t> shape_vec = {500, 10};
  for(std::string ibn : loss_op->input_bns()){
    shape_manager.add_bn_shape_ptr(ibn, new Shape(shape_vec));
  }
  for(std::string obn : loss_op->output_bns()){
    shape_manager.add_bn_shape_ptr(obn, new Shape);
  }
  for(std::string dtbn : loss_op->data_tmp_bns()){
    shape_manager.add_bn_shape_ptr(dtbn, new Shape);
  }
  auto fp = std::bind(&TestShapeFactory::bn2ShapePtr, 
      &shape_manager, std::placeholders::_1)
  loss_op-> InferShape4FwBlobs(fp, kDataParallel, 2, 10);

  Shape* data_shape_ptr = shape_manager.bn2ShapePtr(
      loss_op->input_bns().at(0));
  Shape* loss_shape_ptr = shape_manager.bn2ShapePtr(loss_op->SoleObn());
  Shape* loss_buffer_shape_ptr = shape_manager.bn2ShapePtr(
      loss_op->data_tmp_bns().at(0));
  ASSERT_EQ(*loss_shape_ptr, *data_shape_ptr);
  ASSERT_EQ(*loss_buffer_shape_ptr, *data_shape_ptr);
  */
  TODO();
}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
