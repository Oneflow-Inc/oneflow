#include "relu_op.h"
#include <vector>
#include "gtest/gtest.h"
#include "operator/operator_manager.h"
#include "operator/op_util.h"

namespace oneflow {

TEST(ReluOp, relu_3x5x4) {
  // create relu_op with input shape 3x5x4
  OperatorConf op_conf;
  op_conf.set_name("relu_test");
  op_conf.mutable_relu_conf()->set_in("relu_in");
  op_conf.mutable_relu_conf()->set_out("relu_out");
  auto relu_op = OpMgr::Singleton().ConstructOp(op_conf);
  std::vector<int64_t> input_shape_vec = { 3, 5, 4 };
  HashMap<std::string, Shape*> bn2ShapePtr {
      {relu_op->SoleIbn(), new Shape(input_shape_vec)},
      {relu_op->SoleObn(), new Shape}};
  auto fp = [bn2ShapePtr](const std::string& bn) {
    return bn2ShapePtr.at(bn);
  };
  // do infer shape
  relu_op->InferShape4FwBlobs(fp, kDataParallel, 0, 1);
  // test
  Shape* input_shape_ptr = bn2ShapePtr.at(relu_op->SoleIbn());
  Shape* output_shape_ptr = bn2ShapePtr.at(relu_op->SoleObn());
  ASSERT_EQ(*input_shape_ptr, *output_shape_ptr);
  ASSERT_NE(input_shape_ptr, output_shape_ptr);
}

}  // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
