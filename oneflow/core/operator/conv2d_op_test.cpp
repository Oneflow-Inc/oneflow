#include "oneflow/core/operator/conv2d_op.h"

namespace oneflow {

namespace {

std::shared_ptr<Operator> GetTestConv2dOp() {
  OperatorConf op_conf;
  op_conf.set_name("conv2d_test");
  op_conf.mutable_conv2d_conf()->set_in("conv2d/in");
  op_conf.mutable_conv2d_conf()->set_out("conv2d/out");
  op_conf.mutable_conv2d_conf()->set_out_num(16);
  op_conf.mutable_conv2d_conf()->set_pad_h(4);
  op_conf.mutable_conv2d_conf()->set_pad_w(4);
  op_conf.mutable_conv2d_conf()->set_kernel_size_h(20);
  op_conf.mutable_conv2d_conf()->set_kernel_size_w(20);
  op_conf.mutable_conv2d_conf()->set_stride_h(3);
  op_conf.mutable_conv2d_conf()->set_stride_w(3);
  auto conv2d_op = ConstructOp(op_conf);
  JobConf job_conf;
  job_conf.set_DefaultDataType(DataType::kFloat);
  JobDesc::Singleton()->InitFromJobConf(job_conf);
  return conv2d_op;
}

}  // namespace

TEST(Conv2dOp, dataparallel_conv2d) {
  auto conv2d_op = GetTestConv2dOp();
  HashMap<std::string, BlobDesc*> bn2blob_desc_map{
      {"in", new BlobDesc(Shape({100, 64, 256, 256}), DataType::kFloat, false)},
      {"out", new BlobDesc},
      {"col_buf", new BlobDesc},
      {"filter", new BlobDesc},
      {"bias", new BlobDesc},
      {"bias_multiplier", new BlobDesc}};
  auto Bn2BlobDescFunc = [&bn2blob_desc_map](const std::string& bn) {
    return bn2blob_desc_map.at(bn);
  };
  conv2d_op->InferBlobDescs(Bn2BlobDescFunc, kDataParallel, 0, 1);
  ASSERT_EQ(*Bn2BlobDescFunc("out"),
            BlobDesc(Shape({100, 16, 82, 82}), DataType::kFloat, false));
  ASSERT_EQ(
      *Bn2BlobDescFunc("col_buf"),
      BlobDesc(Shape({100, 82 * 82, 64 * 20 * 20}), DataType::kFloat, false));
  ASSERT_EQ(*Bn2BlobDescFunc("filter"),
            BlobDesc(Shape({16, 64 * 20 * 20}), DataType::kFloat, false));
  ASSERT_EQ(*Bn2BlobDescFunc("bias"),
            BlobDesc(Shape({16}), DataType::kFloat, false));
  ASSERT_EQ(*Bn2BlobDescFunc("bias_multiplier"),
            BlobDesc(Shape({82 * 82}), DataType::kFloat, false));
}

TEST(Conv2dOp, modelparallel_conv2d) {
  auto conv2d_op = GetTestConv2dOp();
  HashMap<std::string, BlobDesc*> bn2shape_ptr{
      {"in", new BlobDesc(Shape({100, 64, 256, 256}), DataType::kFloat, false)},
      {"out", new BlobDesc},
      {"col_buf", new BlobDesc},
      {"filter", new BlobDesc},
      {"bias", new BlobDesc},
      {"bias_multiplier", new BlobDesc}};
  auto Bn2BlobDescFunc = [&bn2shape_ptr](const std::string& bn) {
    return bn2shape_ptr.at(bn);
  };
  conv2d_op->InferBlobDescs(Bn2BlobDescFunc, kModelParallel, 3, 8);
  ASSERT_EQ(*Bn2BlobDescFunc("out"),
            BlobDesc(Shape({100, 2, 82, 82}), DataType::kFloat, false));
  ASSERT_EQ(
      *Bn2BlobDescFunc("col_buf"),
      BlobDesc(Shape({100, 82 * 82, 64 * 20 * 20}), DataType::kFloat, false));
  ASSERT_EQ(*Bn2BlobDescFunc("filter"),
            BlobDesc(Shape({2, 64 * 20 * 20}), DataType::kFloat, false));
  ASSERT_EQ(*Bn2BlobDescFunc("bias"),
            BlobDesc(Shape({2}), DataType::kFloat, false));
  ASSERT_EQ(*Bn2BlobDescFunc("bias_multiplier"),
            BlobDesc(Shape({82 * 82}), DataType::kFloat, false));
}

}  // namespace oneflow
