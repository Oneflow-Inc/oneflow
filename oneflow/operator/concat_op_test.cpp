#include "operator/concat_op.h"
#include "operator/operator_manager.h"
#include "gtest/gtest.h"

namespace oneflow {

TEST(ConcatOp, concat_two_3x3) {
  //create op
  OperatorConf op_conf;
  op_conf.set_name("concat_test");
  op_conf.mutable_concat_conf()->add_in("concat/in0");
  op_conf.mutable_concat_conf()->add_in("concat/in1");
  op_conf.mutable_concat_conf()->set_axis(1);
  op_conf.mutable_concat_conf()->set_out("concat_test_lbn");
  auto concat_op = OpMgr::Singleton().ConstructOp(op_conf);

  std::vector<int64_t> shape_vec = {3, 3};
  HashMap<std::string, Shape*> bn2shape_ptr {
      {concat_op->input_bns().at(0), new Shape(shape_vec)},
      {concat_op->input_bns().at(1), new Shape(shape_vec)},
      {concat_op->SoleObn(), new Shape}};
  auto fp = [&bn2shape_ptr](const std::string& bn) {
    return bn2shape_ptr.at(bn);
  };
  //infershape
  concat_op->InferShape4FwBlobs(fp, kDataParallel, 0, 1);
  //test
  Shape* output_shape_ptr = fp(concat_op->SoleObn());
  ASSERT_EQ(*output_shape_ptr, Shape({3, 6}));
}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
