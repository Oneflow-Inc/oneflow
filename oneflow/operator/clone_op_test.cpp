#include "operator/clone_op.h"
#include "operator/operator_manager.h"
#include "operator/op_util.h"
#include "gtest/gtest.h"

namespace oneflow {

TEST(CloneOp, clone_4x3_3_times) {
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  op_conf.mutable_clone_conf()->set_out_num(3);
  op_conf.mutable_clone_conf()->set_lbn("clone_test_lbn");
  auto clone_op = OpMgr::Singleton().ConstructOp(op_conf);

  TestShapeFactory shape_manager = TestShapeFactory();
  std::vector<int64_t> shape_vec = {4, 3};
  shape_manager.add_bn_shape_ptr(clone_op->SoleIbn(), new Shape(shape_vec));
  for(std::string obn : clone_op->output_bns()){
    shape_manager.add_bn_shape_ptr(obn, new Shape);
  }

  auto fp = std::bind(&TestShapeFactory::bn2ShapePtr, 
      &shape_manager, std::placeholders::_1)
  clone_op->InferShape4FwBlobs(fp, kDataParallel, 3, 10);

  Shape* input_shape_ptr = shape_manager.bn2ShapePtr(clone_op->SoleIbn());
  for(std::string obn : clone_op->output_bns()){
    ASSERT_EQ(*input_shape_ptr, *shape_manager.bn2ShapePtr(obn));
    ASSERT_NE(input_shape_ptr, shape_manager.bn2ShapePtr(obn));
  }

}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
