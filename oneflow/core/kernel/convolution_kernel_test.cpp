#include "oneflow/core/kernel/convolution_kernel.h"
#include "oneflow/core/kernel/kernel_test_common.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobFunc(
    bool has_bias_term) {
  using KTC = KTCommon<device_type, T>;
  auto bn2blob = new HashMap<std::string, Blob*>;
  (*bn2blob)["in"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 1, 3, 3}), GetDataType<T>::val, false),
      {1, -1, 2, 3, 1, -1, 2, 1, -2});
  (*bn2blob)["col_buf"] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({1, 4, 4}), GetDataType<T>::val, false));
  (*bn2blob)["weight"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 4}), GetDataType<T>::val, false),
      {0.1f, 0.2f, 0.2f, 0.4f, -0.3f, 0.1f, 0.1f, 0.2f});
  (*bn2blob)["out"] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({1, 2, 2, 2}), GetDataType<T>::val, false));
  (*bn2blob)["bias"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2}), GetDataType<T>::val, false), {0, 0});
  (*bn2blob)[GenDiffBn("out")] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 2, 2, 2}), GetDataType<T>::val, false),
      {1, 3, 2, 2, 2, 1, 1, 1});
  (*bn2blob)[GenDiffBn("in")] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({1, 1, 3, 3}), GetDataType<T>::val, false));
  (*bn2blob)[GenDiffBn("weight")] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({2, 4}), GetDataType<T>::val, false));
  (*bn2blob)[GenDiffBn("bias")] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({2}), GetDataType<T>::val, false));
  (*bn2blob)["bias_multiplier"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({4}), GetDataType<T>::val, false), {1, 1, 1, 1});
  (*bn2blob)["expected_out"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 2, 2, 2}), GetDataType<T>::val, false),
      {0.9f, 0.1f, 1.3f, -0.7f, 0.1f, 0.4f, -0.4f, -0.7f});
  (*bn2blob)["expected_bias_diff"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2}), GetDataType<T>::val, false), {2, 1.25f});
  (*bn2blob)["expected_in_diff"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 1, 3, 3}), GetDataType<T>::val, false),
      {-0.5f, 0.4f, 0.7f, 0.3f, 1.9f, 1.9f, 0.5f, 1.5f, 1.0f});
  (*bn2blob)["expected_weight_diff"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 4}), GetDataType<T>::val, false),
      {1.5f, 1.25f, 3, -1, 1.25f, 0, 2.5f, 0});
  (*bn2blob)["fwd_workspace"] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({1, 1}), GetDataType<T>::val, false));
  (*bn2blob)["bwd_weight_workspace"] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({1, 1}), GetDataType<T>::val, false));
  (*bn2blob)["bwd_data_workspace"] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({1, 1}), GetDataType<T>::val, false));
  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

template<DeviceType device_type, typename T>
Kernel* BuildConvolutionKernel(bool has_bias_term) {
  OperatorConf op_conf;
  op_conf.set_name("convolution_test");
  op_conf.mutable_convolution_conf()->set_in("convolution/in");
  op_conf.mutable_convolution_conf()->set_out("convolution/out");
  op_conf.mutable_convolution_conf()->set_out_num(1);
  op_conf.mutable_convolution_conf()->set_pad_h(0);
  op_conf.mutable_convolution_conf()->set_pad_w(0);
  op_conf.mutable_convolution_conf()->set_kernel_h(2);
  op_conf.mutable_convolution_conf()->set_kernel_w(2);
  op_conf.mutable_convolution_conf()->set_stride_h(1);
  op_conf.mutable_convolution_conf()->set_stride_w(1);
  op_conf.mutable_convolution_conf()->set_dilation_h(1);
  op_conf.mutable_convolution_conf()->set_dilation_w(1);
  op_conf.mutable_convolution_conf()->set_has_bias_term(has_bias_term);
  op_conf.mutable_convolution_conf()->set_cudnn_fwd_algo(0);
  op_conf.mutable_convolution_conf()->set_cudnn_bwd_weight_algo(0);
  op_conf.mutable_convolution_conf()->set_cudnn_bwd_data_algo(0);

  auto convolution_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  convolution_op->ToProto(&op_proto);
  auto conv_kernel = new CudnnConvolutionKernel<device_type, T>();
  conv_kernel->InitFromOpProto(op_proto);
  return conv_kernel;
}

template<DeviceType device_type, typename T>
void TestConvolutionKernel(bool has_bias_term) {
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);
  auto BnInOp2BlobFunc = BuildBnInOp2BlobFunc<device_type, T>(has_bias_term);
  auto conv_kernel = BuildConvolutionKernel<device_type, T>(has_bias_term);

  conv_kernel->Forward(ctx, BnInOp2BlobFunc);
  conv_kernel->Backward(ctx, BnInOp2BlobFunc);

  SyncStream<device_type>(&ctx);

  KTCommon<device_type, T>::CheckResult(BnInOp2BlobFunc, "out", "expected_out");

  KTCommon<device_type, T>::CheckResult(BnInOp2BlobFunc, GenDiffBn("in"),
                                        "expected_in_diff");

  KTCommon<device_type, T>::CheckResult(BnInOp2BlobFunc, GenDiffBn("weight"),
                                        "expected_weight_diff");

  if (has_bias_term) {
    KTCommon<device_type, T>::CheckResult(BnInOp2BlobFunc, GenDiffBn("bias"),
                                          "expected_bias_diff");
  }
}

}  // namespace

TEST(ConvKernel, cudnn_conv_kernel) {
  test::TestConvolutionKernel<DeviceType::kGPU, float>(true);
}

}  // namespace test

}  // namespace oneflow
