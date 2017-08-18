#include "oneflow/core/operator/clone_op.h"

namespace oneflow {

TEST(CloneOp, clone_4x3_3_times) {
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  op_conf.mutable_clone_conf()->set_out_num(3);
  op_conf.mutable_clone_conf()->set_lbn("clone_test_lbn");
  auto clone_op = ConstructOp(op_conf);

  HashMap<std::string, Shape*> bn2shape_ptr{
      {clone_op->SoleIbn(), new Shape({4, 3})}};
  for (std::string obn : clone_op->output_bns()) {
    bn2shape_ptr.emplace(obn, new Shape);
  }
  auto fp = [&bn2shape_ptr](const std::string& bn) {
    return bn2shape_ptr.at(bn);
  };

  clone_op->InferBlobDesc4FwBlobs(fp, kDataParallel, 3, 10);

  Shape* input_shape_ptr = bn2shape_ptr.at(clone_op->SoleIbn());
  for (std::string obn : clone_op->output_bns()) {
    ASSERT_EQ(*input_shape_ptr, *bn2shape_ptr.at(obn));
    ASSERT_NE(input_shape_ptr, bn2shape_ptr.at(obn));
  }
}

}  // namespace oneflow
