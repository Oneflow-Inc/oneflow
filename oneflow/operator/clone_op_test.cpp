#include "operator/clone_op.h"
#include "operator/operator_manager.h"
#include "gtest/gtest.h"

namespace oneflow {

HaspMap<std::string, Shape*> bn_in_op2shape_ptr;

Shape* GetShapePtr4BnInOp(const std::string& bn){
  return bn_in_op2shape_ptr[bn];
}

TEST(CloneOp, clone_4x3_3_times) {
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  op_conf.mutable_clone_conf()->set_out_num(3);
  op_conf.mutable_clone_conf()->set_lbn("clone_test_lbn");
  auto clone_op = OpMgr::Singleton().ConstructOp(op_conf);

  bn_in_op2shape_ptr.clear();
  std::vector<int64_t> shape_vec = {4, 3};
  bn_in_op2shape_ptr[clone_op->SoleIbn()] = new Shape(shape_vec);
  for(std::string obn : clone_op->output_bns()){
    bn_in_op2shape_ptr[obn] = new Shape;
  }

  clone_op->InferShape4FwBlobs(GetShapePtr4BnInOp, kDataParallel, 3, 10);

  Shape* input_shape_ptr = GetShapePtrPtr4BnInOp(clone_op->SoleIbn());
  for(std::string obn : clone_op->output_bns()){
    ASSERT_EQ(*input_shape_ptr, *GetShapePtrPtr4BnInOp(obn));
    ASSERT_NE(input_shape_ptr, GetShapePtrPtr4BnInOp(obn));
  }

}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
