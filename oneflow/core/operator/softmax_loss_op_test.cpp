#include "oneflow/core/operator/softmax_loss_op.h"

namespace oneflow {

TEST(SoftmaxLossOp, softmax_loss_3x5) {
  // create softmax_loss_op
  OperatorConf op_conf;
  op_conf.set_name("softmax_loss_test");
  op_conf.mutable_softmax_loss_conf()->set_in("softmax_loss/in");
  op_conf.mutable_softmax_loss_conf()->set_label("softmax_loss/label");
  op_conf.mutable_softmax_loss_conf()->set_loss("softmax_loss/loss");
  auto softmax_loss_op = ConstructOp(op_conf);
  HashMap<std::string, Shape*> bn2shape_ptr{{"in", new Shape({3, 5})},
                                            {"label", new Shape({3, 5})},
                                            {"prob", new Shape},
                                            {"tmp_1D", new Shape},
                                            {"loss", new Shape}};
  auto fp = [&bn2shape_ptr](const std::string& bn) {
    return bn2shape_ptr.at(bn);
  };
  // infershape
  softmax_loss_op->InferShape4FwBlobs(fp, kDataParallel, 0, 1);
  // test
  ASSERT_EQ(*fp("loss"), Shape({1}));
  ASSERT_EQ(*fp("prob"), Shape({3, 5}));
  ASSERT_EQ(*fp("tmp_1D"), Shape({3}));
}

}  // namespace oneflow
