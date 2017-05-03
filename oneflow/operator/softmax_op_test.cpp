#include "operator/softmax_op.h"
#include "operator/operator_manager.h"
#include "gtest/gtest.h"

namespace oneflow {

TEST(SoftmaxOp, softmax_3x4x5) {
  /*
  OperatorConf op_conf;
  op_conf.set_name("softmax_test");
  op_conf.mutable_softmax_conf()->set_axis(1);
  op_conf.mutable_softmax_conf()->set_in("softmax/in");
  op_conf.mutable_softmax_conf()->set_out("softmax/out");
  auto softmax_op = OpMgr::Singleton().ConstructOp(op_conf);

  std::vector<int64_t> shape_vec = {3, 4, 5};
  TestShapeFactory shape_factory = TestShapeFactory();
  shape_factory.add_bn_shape_ptr(softmax_op->SoleIbn, new Shape(shape_vec));
  shape_factory.add_bn_shape_ptr(softmax_op->SoleObn, new Shape);
  auto fp = std::bind(&TestShapeFactory::bn2ShapePtr,
                      &shape_factory,
                      std::placeholders::_1);
  softmax_op->InferShape4FwBlobs(fp, kDataParallel, 0, 1);

  Shape* input_shape_ptr = shape_factory.bn2ShapePtr(softmax_op->SoleIbn());
  Shape* output_shape_ptr = shape_factory.bn2ShapePtr(softmax_op->SoleObn());
  
  ASSERT_EQ(output_shape_ptr->NumAxes(), input_shape_ptr->NumAxes() - 1);
  for (int64_t i = 0; i < input_shape_ptr->NumAxes(); ++i) {
    if (i == 1) continue;
    ASSERT_EQ(output_shape_ptr->At(i > 1 ? i - 1 : i), input_shape_ptr->At(i));
  }
  */
  TODO();
}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
