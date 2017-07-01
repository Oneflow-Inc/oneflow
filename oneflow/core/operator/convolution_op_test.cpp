#include "oneflow/core/operator/convolution_op.h"

namespace oneflow {

TEST(ConvolutionOp, TestForInferShape4FwBlobs) {
  // create conv_op
  OperatorConf op_conf;
  op_conf.set_name("convolution_test");
  op_conf.mutable_convolution_conf()->set_in("convolution/in");
  op_conf.mutable_convolution_conf()->set_out("convolution/out");
  op_conf.mutable_convolution_conf()->set_out_num(16);
  op_conf.mutable_convolution_conf()->add_pad(4);
  op_conf.mutable_convolution_conf()->add_pad(4);
  op_conf.mutable_convolution_conf()->add_kernel_size(20);
  op_conf.mutable_convolution_conf()->add_kernel_size(20);
  op_conf.mutable_convolution_conf()->add_stride(3);
  op_conf.mutable_convolution_conf()->add_stride(3);
  auto convolution_op = OpMgr::Singleton().ConstructOp(op_conf);
  std::vector<int64_t> input_vec = {100, 64, 256, 256};
  HashMap<std::string, Shape*> bn2shape_ptr{
      {convolution_op->SoleIbn(), new Shape(input_vec)},
      {convolution_op->SoleObn(), new Shape},
      {convolution_op->data_tmp_bns().at(0), new Shape},
      {convolution_op->model_bns().at(0), new Shape},
      {convolution_op->model_bns().at(1), new Shape},
      {convolution_op->model_tmp_bns().at(0), new Shape}};
  auto fp = [&bn2shape_ptr](const std::string& bn) {
    return bn2shape_ptr.at(bn);
  };
  // infershape
  convolution_op->InferShape4FwBlobs(fp, kDataParallel, 0, 1);
  // test
  Shape* output_shape_ptr = fp(convolution_op->SoleObn());
  Shape* colbuf_shape_ptr = fp(convolution_op->data_tmp_bns().at(0));
  Shape* weight_shape_ptr = fp(convolution_op->model_bns().at(0));
  Shape* bias_shape_ptr = fp(convolution_op->model_bns().at(1));
  Shape* biasmult_shape_ptr = fp(convolution_op->model_tmp_bns().at(0));
  ASSERT_EQ(*output_shape_ptr, Shape({100, 16, 82, 82}));
  ASSERT_EQ(*colbuf_shape_ptr, Shape({100, 82 * 82, 64 * 20 * 20}));
  ASSERT_EQ(*weight_shape_ptr, Shape({16, 64 * 20 * 20}));
  ASSERT_EQ(*bias_shape_ptr, Shape({16}));
  ASSERT_EQ(*biasmult_shape_ptr, Shape({82 * 82}));
}

}  // namespace oneflow
