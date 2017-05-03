#include <utility>
#include <string>
#include <vector>
#include "operator/multinomial_logistic_loss_op.h"
#include "operator/operator_manager.h"
#include "gtest/gtest.h"

namespace oneflow {


TEST(MultinomialLogisticLossOp, test_loss_op) {
  OperatorConf op_conf;
  op_conf.set_name("multinomial_logistic_loss_op_test");
  op_conf.mutable_multinomial_logistic_loss_conf()->set_data("data");
  op_conf.mutable_multinomial_logistic_loss_conf()->set_label("label");
  op_conf.mutable_multinomial_logistic_loss_conf()->set_loss("loss");
  auto loss_op = OpMgr::Singleton().ConstructOp(op_conf);

  HashMap<std::string, Shape*> bn2shape_ptr;
  std::vector<int64_t> shape_vec = {500, 10};
  for (std::string ibn : loss_op->input_bns()) {
    bn2shape_ptr.insert(std::make_pair(ibn, new Shape(shape_vec)));
  }
  for (std::string obn : loss_op->output_bns()) {
    bn2shape_ptr.insert(std::make_pair(obn, new Shape));
  }
  for (std::string dtbn : loss_op->data_tmp_bns()) {
    bn2shape_ptr.insert(std::make_pair(dtbn, new Shape));
  }
  auto fp = [&bn2shape_ptr](const std::string& bn) {
    return bn2shape_ptr.at(bn);
  };

  loss_op-> InferShape4FwBlobs(fp, kDataParallel, 2, 10);

  Shape* data_shape_ptr = bn2shape_ptr.at(loss_op->input_bns().at(0));
  Shape* loss_shape_ptr = bn2shape_ptr.at(loss_op->SoleObn());
  Shape* loss_buffer_shape_ptr = bn2shape_ptr.at(
      loss_op->data_tmp_bns().at(0));
  ASSERT_EQ(*loss_shape_ptr, *data_shape_ptr);
  ASSERT_EQ(*loss_buffer_shape_ptr, *data_shape_ptr);
}

}  // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
