#include "operator/clone_op.h"
#include "operator/operator_manager.h"
#include "gtest/gtest.h"

namespace oneflow {

TEST(CloneOp, clone_4x3_3_times) {
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  op_conf.mutable_clone_conf()->set_out_num(3);
  op_conf.mutable_clone_conf()->set_lbn("clone_test_lbn");
  auto clone_op = OpMgr::Singleton().ConstructOp(op_conf);

  std::vector<int64_t> shape_vec = {4, 3};
  clone_op->SetShapePtr(clone_op->SoleIbn(), new Shape(shape_vec));
  for(std::string obn : clone_op->output_bns()){
    clone_op->SetShapePtr(obn, new Shape);
  }

  clone_op->InferShape4ObAndDtbFromIb();

  Shape* input_shape_ptr = clone_op->GetShapePtr(clone_op->SoleIbn());
  for(std::string obn : clone_op->output_bns()){
    ASSERT_TRUE(*input_shape_ptr == *(clone_op->GetShapePtr(obn)));
    ASSERT_TRUE(input_shape_ptr != clone_op->GetShapePtr(obn));
  }

}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
