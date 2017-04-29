#include "operator/softmax_op.h"
#include "operator/operator_manager.h"
#include "gtest/gtest.h"

namespace oneflow {

TEST(SoftmaxOp, softmax_3x4x5) {
  OperatorConf op_conf;
  op_conf.set_name("softmax_test");
  op_conf.mutable_softmax_conf()->set_axis(1);
  op_conf.mutable_softmax_conf()->set_in("softmax/in");
  op_conf.mutable_softmax_conf()->set_out("softmax/out");
  auto softmax_op = OpMgr::Singleton().ConstructOp(op_conf);

  std::vector<int64_t> shape_vec = {3, 4, 5};
  softmax_op->SetShapePtr(softmax_op->SoleIbn(), new Shape(shape_vec)); 
  softmax_op->SetShapePtr(softmax_op->SoleObn(), new Shape);
  
  softmax_op->InferShape4ObAndDtbFromIb();

  Shape* input_shape_ptr = softmax_op->GetShapePtr(softmax_op->SoleIbn());
  Shape* output_shape_ptr = softmax_op->GetShapePtr(softmax_op->SoleObn());
  
  ASSERT_TRUE(output_shape_ptr->NumAxes() == input_shape_ptr->NumAxes() - 1);
  for (int i = 0; i < input_shape_ptr->NumAxes(); ++i) {
    if (i == 1) continue;
    ASSERT_TRUE(output_shape_ptr->At(i>1?i-1:i) == input_shape_ptr->At(i));
  }
}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
