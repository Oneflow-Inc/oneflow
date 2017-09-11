#include "oneflow/core/operator/convolution_op.h"

namespace oneflow {

namespace {

std::shared_ptr<Operator> GetTestConvolutionOp() {
  OperatorConf op_conf;
  op_conf.set_name("convolution_test");
  op_conf.mutable_convolution_conf()->set_in("convolution/in");
  op_conf.mutable_convolution_conf()->set_out("convolution/out");
  op_conf.mutable_convolution_conf()->set_has_bias_term(true);
  op_conf.mutable_convolution_conf()->set_out_num(16);
  op_conf.mutable_convolution_conf()->set_pad_h(4);
  op_conf.mutable_convolution_conf()->set_pad_w(4);
  op_conf.mutable_convolution_conf()->set_kernel_size_h(20);
  op_conf.mutable_convolution_conf()->set_kernel_size_w(20);
  op_conf.mutable_convolution_conf()->set_stride_h(3);
  op_conf.mutable_convolution_conf()->set_stride_w(3);
  auto convolution_op = ConstructOp(op_conf);
  JobConf job_conf;
  job_conf.set_default_data_type(DataType::kFloat);
  JobDesc::Singleton()->InitFromJobConf(job_conf);
  return convolution_op;
}

}  // namespace

TEST(ConvolutionOp, dataparallel_convolution) {
  auto convolution_op = GetTestConvolutionOp();
  HashMap<std::string, BlobDesc*> bn2blob_desc_map{
      {"in", new BlobDesc(Shape({100, 64, 256, 256}), DataType::kFloat, false)},
      {"out", new BlobDesc},
      {"col_buf", new BlobDesc},
      {"weight", new BlobDesc},
      {"bias", new BlobDesc},
      {"bias_multiplier", new BlobDesc}};
  auto Bn2BlobDescFunc = [&bn2blob_desc_map](const std::string& bn) {
    return bn2blob_desc_map.at(bn);
  };
  convolution_op->InferBlobDesc4FwBlobs(Bn2BlobDescFunc, kDataParallel, 0, 1);
  ASSERT_EQ(*Bn2BlobDescFunc("out"),
            BlobDesc(Shape({100, 16, 82, 82}), DataType::kFloat, false));
  ASSERT_EQ(
      *Bn2BlobDescFunc("col_buf"),
      BlobDesc(Shape({100, 82 * 82, 64 * 20 * 20}), DataType::kFloat, false));
  ASSERT_EQ(*Bn2BlobDescFunc("weight"),
            BlobDesc(Shape({16, 64 * 20 * 20}), DataType::kFloat, false));
  ASSERT_EQ(*Bn2BlobDescFunc("bias"),
            BlobDesc(Shape({16}), DataType::kFloat, false));
  ASSERT_EQ(*Bn2BlobDescFunc("bias_multiplier"),
            BlobDesc(Shape({82 * 82}), DataType::kFloat, false));
}

TEST(ConvolutionOp, modelparallel_convolution) {
  auto convolution_op = GetTestConvolutionOp();
  HashMap<std::string, BlobDesc*> bn2shape_ptr{
      {"in", new BlobDesc(Shape({100, 64, 256, 256}), DataType::kFloat, false)},
      {"out", new BlobDesc},
      {"col_buf", new BlobDesc},
      {"weight", new BlobDesc},
      {"bias", new BlobDesc},
      {"bias_multiplier", new BlobDesc}};
  auto Bn2BlobDescFunc = [&bn2shape_ptr](const std::string& bn) {
    return bn2shape_ptr.at(bn);
  };
  convolution_op->InferBlobDesc4FwBlobs(Bn2BlobDescFunc, kModelParallel, 3, 8);
  ASSERT_EQ(*Bn2BlobDescFunc("out"),
            BlobDesc(Shape({100, 2, 82, 82}), DataType::kFloat, false));
  ASSERT_EQ(
      *Bn2BlobDescFunc("col_buf"),
      BlobDesc(Shape({100, 82 * 82, 64 * 20 * 20}), DataType::kFloat, false));
  ASSERT_EQ(*Bn2BlobDescFunc("weight"),
            BlobDesc(Shape({2, 64 * 20 * 20}), DataType::kFloat, false));
  ASSERT_EQ(*Bn2BlobDescFunc("bias"),
            BlobDesc(Shape({2}), DataType::kFloat, false));
  ASSERT_EQ(*Bn2BlobDescFunc("bias_multiplier"),
            BlobDesc(Shape({82 * 82}), DataType::kFloat, false));
}

}  // namespace oneflow
