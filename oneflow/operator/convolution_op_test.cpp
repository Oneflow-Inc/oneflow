#include "operator/convolution_op.h"
#include "operator/operator_manager.h"
#include "gtest/gtest.h"

namespace oneflow {

TEST(ConvolutionOp, TestForInferShape4FwBlobs) {
  OperatorConf op_conf;
  op_conf.set_name("convolution_test");
  op_conf.mutable_convolution_conf()->set_in("convolution/in");
  op_conf.mutable_convolution_conf()->set_out("convolution/out");
  op_conf.mutable_convolution_conf()->set_out_num(16);
  op_conf.mutable_convolution_conf()->set_pad(4);
  op_conf.mutable_convolution_conf()->set_kernel_size(20);
  op_conf.mutable_convolution_conf()->set_stride(3);
  auto convolution_op = OpMgr::Singleton().ConstructOp(op_conf);

  std::vector<int64_t> input_vec = {100, 64, 256, 256};
  TestShapeFactory shape_factory = TestShapeFactory();
  shape_factory.add_bn_shape_ptr(convolution_op->SoleIbn(), new Shape(input_vec));
  shape_factory.add_bn_shape_ptr(convolution_op->SoleObn(), new Shape);
  shape_factory.add_bn_shape_ptr(convolution_op->data_tmp_bns().at(0), new Shape);
  shape_factory.add_bn_shape_ptr(convolution_op->model_bns().at(0), new Shape);
  shape_factory.add_bn_shape_ptr(convolution_op->model_bns().at(1), new Shape);
  shape_factory.add_bn_shape_ptr(convolution_op->model_bns().at(0), new Shape);
  
  auto fp = std::bind(&TestShapeFactory::bn2ShapePtr,
                      &shape_factory,
                      std::placeholders::_1); 
  convolution_op->InferShape4FwBlobs(fp, kDataParallel, 0, 1);

  Shape* input_shape_ptr = shape_factory.bn2ShapePtr(convolution_op->SoleIbn());
  Shape* output_shape_ptr = shape_factory.bn2ShapePtr(convolution_op->SoleObn());
  Shape* colbuf_shape_ptr = shape_factory.bn2ShapePtr(convolution_op->data_tmp_bns().at(0));
  Shape* weight_shape_ptr = shape_factory.bn2ShapePtr(convolution_op->model_bns().at(0));
  Shape* bias_shape_ptr = shape_factory.bn2ShapePtr(convolution_op->model_bns().at(1));
  Shape* biasmult_shape_ptr = shape_factory.bn2ShapePtr(convolution_op->model_tmp_bns().at(0));

  ASSERT_EQ(*output_shape_ptr, Shape({100, 16, 82, 82}));
  ASSERT_EQ(*colbuf_shape_ptr, Shape({100, 82 * 82, 64 * 20 * 20}));
  ASSERT_EQ(*weight_shape_ptr, Shape({16, 64 * 20 * 20}));
  ASSERT_EQ(*bias_shape_ptr, Shape({16}));
  ASSERT_EQ(*biasmult_shape_ptr, Shape({82 * 82}));
}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
