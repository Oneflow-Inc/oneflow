#include "oneflow/core/kernel/convolution_kernel.h"
#include "oneflow/core/kernel/kernel_test_common.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr(
    bool has_bias_term) {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  FloatingPointType in_mat[] = {1, -1, 2, 3, 1, -1, 2, 1, -2};  // 1x1x3x3
  FloatingPointType weight_mat[] = {0.1,  0.2, 0.2, 0.4,
                                    -0.3, 0.1, 0.1, 0.2};  // 2x(1*2*2)
  FloatingPointType bias_mat[2] = {0};                     // 1
  FloatingPointType bias_multiplier_mat[] = {1, 1, 1, 1};  // 2*2
  FloatingPointType out_mat[8] = {100};                    // 1x2x2x2
  FloatingPointType in_diff_mat[9] = {10000};              // 1x1x3x3
  FloatingPointType weight_diff_mat[8] = {40};             // 2x(1*2*2)
  FloatingPointType expected_weight_diff_mat[] = {1.5, 1.25, 3, -1, 1.25,
                                                  0,   2.5,  0};  // 2x(1*2*2)
  FloatingPointType bias_diff_mat[2] = {0};                       // 2
  FloatingPointType expected_bias_diff_mat[] = {2, 1.25};       // 2
  FloatingPointType out_diff_mat[] = {1, 3, 2, 2, 2, 1, 1, 1};  // 1x2x2x2
  FloatingPointType expected_out_mat[] = {0.9, 0.1, 1.3,  -0.7,
                                          0.1, 0.4, -0.4, -0.7};  // 1x2x2x2
  FloatingPointType expected_in_diff_mat[] = {-0.5, 0.4, 0.7, 0.3, 1.9,
                                              1.9,  0.5, 1.5, 1.0};

  FloatingPointType col_buf_mat[16] = {0};

  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  (*bn2blob_ptr)["in"] = KTCommon::CreateBlobWithVector({1, 1, 3, 3}, in_mat);
  (*bn2blob_ptr)["col_buf"] =
      KTCommon::CreateBlobWithVector({1, 4, 4}, col_buf_mat);
  (*bn2blob_ptr)["weight"] = KTCommon::CreateBlobWithVector({2, 4}, weight_mat);
  (*bn2blob_ptr)["out"] = KTCommon::CreateBlobWithVector({1, 2, 2, 2}, out_mat);
  (*bn2blob_ptr)["out_diff"] =
      KTCommon::CreateBlobWithVector({1, 2, 2, 2}, out_diff_mat);
  (*bn2blob_ptr)["in_diff"] =
      KTCommon::CreateBlobWithVector({1, 1, 3, 3}, in_diff_mat);
  (*bn2blob_ptr)["weight_diff"] =
      KTCommon::CreateBlobWithVector({2, 4}, weight_diff_mat);

  (*bn2blob_ptr)["bias"] = KTCommon::CreateBlobWithVector({2}, bias_mat);
  (*bn2blob_ptr)["bias_multiplier"] =
      KTCommon::CreateBlobWithVector({4}, bias_multiplier_mat);
  (*bn2blob_ptr)["bias_diff"] =
      KTCommon::CreateBlobWithVector({2}, bias_diff_mat);
  (*bn2blob_ptr)["expected_bias_diff"] =
      KTCommon::CreateBlobWithVector({2}, expected_bias_diff_mat);
  (*bn2blob_ptr)["expected_out"] =
      KTCommon::CreateBlobWithVector({1, 2, 2, 2}, expected_out_mat);
  (*bn2blob_ptr)["expected_in_diff"] =
      KTCommon::CreateBlobWithVector({1, 1, 3, 3}, expected_in_diff_mat);
  (*bn2blob_ptr)["expected_weight_diff"] =
      KTCommon::CreateBlobWithVector({2, 4}, expected_weight_diff_mat);

  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildConvolutionKernel(bool has_bias_term) {
  OperatorConf op_conf;
  op_conf.set_name("convolution_test");
  op_conf.mutable_convolution_conf()->set_in("convolution/in");
  op_conf.mutable_convolution_conf()->set_out("convolution/out");
  op_conf.mutable_convolution_conf()->set_out_num(1);
  op_conf.mutable_convolution_conf()->add_pad(0);
  op_conf.mutable_convolution_conf()->add_pad(0);
  op_conf.mutable_convolution_conf()->add_kernel_size(2);
  op_conf.mutable_convolution_conf()->add_kernel_size(2);
  op_conf.mutable_convolution_conf()->add_stride(1);
  op_conf.mutable_convolution_conf()->add_stride(1);
  op_conf.mutable_convolution_conf()->add_dilation(1);
  op_conf.mutable_convolution_conf()->add_dilation(1);
  op_conf.mutable_convolution_conf()->set_has_bias_term(has_bias_term);
  auto convolution_op = ConstructOp(op_conf);

  OperatorProto op_proto;
  convolution_op->ToProto(&op_proto);

  auto conv_kernel = new ConvolutionKernel<device_type, FloatingPointType>();
  conv_kernel->InitFromOpProto(op_proto);

  return conv_kernel;
}

template<DeviceType device_type, typename FloatingPointType>
void TestConvolutionKernel(bool has_bias_term) {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);
  auto BnInOp2BlobPtr =
      BuildBnInOp2BlobPtr<device_type, FloatingPointType>(has_bias_term);
  auto conv_kernel =
      BuildConvolutionKernel<device_type, FloatingPointType>(has_bias_term);

  conv_kernel->Forward(ctx, BnInOp2BlobPtr);
  conv_kernel->Backward(ctx, BnInOp2BlobPtr);

  KTCommon::SyncStream(&ctx);

  KTCommon::CheckResult(BnInOp2BlobPtr, "out", "expected_out");
  KTCommon::CheckResult(BnInOp2BlobPtr, "in_diff", "expected_in_diff");
  if (has_bias_term) {
    KTCommon::CheckResult(BnInOp2BlobPtr, "weight_diff",
                          "expected_weight_diff");
    KTCommon::CheckResult(BnInOp2BlobPtr, "bias_diff", "expected_bias_diff");
  }
}

}  // namespace

TEST(ConvKernel, conv_kernel_cpu) {
  test::TestConvolutionKernel<DeviceType::kCPU, float>(true);
  test::TestConvolutionKernel<DeviceType::kCPU, double>(true);
  test::TestConvolutionKernel<DeviceType::kGPU, float>(true);
  test::TestConvolutionKernel<DeviceType::kGPU, double>(true);
}

}  // namespace test

}  // namespace oneflow
