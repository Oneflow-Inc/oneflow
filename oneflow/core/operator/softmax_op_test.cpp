#include "oneflow/core/operator/softmax_op.h"

namespace oneflow {

TEST(SoftmaxOp, softmax_3x5) {
  // create softmax_op
  OperatorConf op_conf;
  op_conf.set_name("softmax_test");
  op_conf.mutable_softmax_conf()->set_in("softmax/in");
  op_conf.mutable_softmax_conf()->set_out("softmax/out");
  auto softmax_op = ConstructOp(op_conf);
  HashMap<std::string, Shape*> bn2shape_ptr{
      {softmax_op->SoleIbn(), new Shape({3, 5})},
      {softmax_op->SoleObn(), new Shape},
      {softmax_op->SoleDtbn(), new Shape}};
  auto fp = [&bn2shape_ptr](const std::string& bn) {
    return bn2shape_ptr.at(bn);
  };
  // infershape
  softmax_op->InferBlobDesc4FwBlobs(fp, kDataParallel, 0, 1);
  // test
  Shape* output_shape_ptr = fp(softmax_op->SoleObn());
  Shape* tmp_shape_ptr = fp(softmax_op->SoleDtbn());
  ASSERT_EQ(*output_shape_ptr, Shape({3, 5}));
  ASSERT_EQ(*tmp_shape_ptr, Shape({3}));
}

}  // namespace oneflow
