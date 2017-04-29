#include "relu_op.h"
#include "gtest/gtest.h"
#include "operator/operator_manager.h"

namespace oneflow {

TEST(ReluOp, relu_3x5x4) {
  // create relu_op with input shape 3x5x4
  OperatorConf op_conf;
  op_conf.set_name("relu_test");
  op_conf.mutable_relu_conf()->set_in("relu_in");
  op_conf.mutable_relu_conf()->set_out("relu_out");
  auto relu_op = OpMgr::Singleton().ConstructOp(op_conf);
  std::vector<int64_t> shape_vec = { 3, 5, 4 };
  relu_op->SetShapePtr(relu_op->SoleIbn(), new Shape(shape_vec));
  relu_op->SetShapePtr(relu_op->SoleObn(), new Shape);
  // do infer shape
  relu_op->InferShape4ObAndDtbFromIb();
  // test
  Shape* input_shape_ptr = relu_op->GetShapePtr(relu_op->SoleIbn());
  Shape* output_shape_ptr = relu_op->GetShapePtr(relu_op->SoleObn());
  ASSERT_EQ(*input_shape_ptr, *output_shape_ptr);
  ASSERT_NE(input_shape_ptr, output_shape_ptr);
}

}// namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}