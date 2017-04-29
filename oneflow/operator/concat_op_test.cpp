#include "operator/concat_op.h"
#include "operator/operator_manager.h"
#include "gtest/gtest.h"

namespace oneflow {

TEST(ConcatOp, concat_two_3x3) {
  OperatorConf op_conf;
  op_conf.set_name("concat_test");
  op_conf.mutable_concat_conf()->add_in("concat/in0");
  op_conf.mutable_concat_conf()->add_in("concat/in1");
  op_conf.mutable_concat_conf()->set_axis(1);
  op_conf.mutable_concat_conf()->set_out("concat_test_lbn");
  auto concat_op = OpMgr::Singleton().ConstructOp(op_conf);

  std::vector<int64_t> shape_vec = {3, 3};
  concat_op->SetShapePtr(concat_op->SoleObn(), new Shape);
  for(std::string ibn : concat_op->input_bns()) {
    concat_op->SetShapePtr(ibn, new Shape(shape_vec));
  }

  concat_op->InferShape4ObAndDtbFromIb();

  Shape* output_shape_ptr = concat_op->GetShapePtr(concat_op->SoleObn());
  int concat_sum = 0;
  for(std::string ibn : concat_op->input_bns()) {
    ASSERT_TRUE(output_shape_ptr->At(0) == concat_op->GetShapePtr(ibn)->At(0));
    concat_sum += concat_op->GetShapePtr(ibn)->At(1);
  }
  ASSERT_TRUE(output_shape_ptr->At(1) == concat_sum);
}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
