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
  op_conf.mutable_convolution_conf()->set_kernel_size_h(2);
  op_conf.mutable_convolution_conf()->set_kernel_size_w(2);
  op_conf.mutable_convolution_conf()->set_stride_h(1);
  op_conf.mutable_convolution_conf()->set_stride_w(1);
  op_conf.mutable_convolution_conf()->set_dilation_h(1);
  op_conf.mutable_convolution_conf()->set_dilation_w(1);
  op_conf.mutable_convolution_conf()->set_has_bias_term(has_bias_term);
  auto convolution_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  convolution_op->ToProto(&op_proto);
  auto conv_kernel = new ConvolutionKernel<device_type, T>();
  conv_kernel->InitFromOpProto(op_proto);
  return conv_kernel;
}

template<DeviceType device_type, typename T>
void TestConvolutionKernel(bool has_bias_term) {
  JobConf job_conf;
  job_conf.set_DefaultDataType(GetDataType<T>::val);
  JobDesc::Singleton()->InitFromJobConf(job_conf);
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

TEST(ConvKernel, conv_kernel_cpu) {
#define MAKE_ENTRY(x, y, z) \
  test::TestConvolutionKernel<x, OF_PP_PAIR_FIRST(y)>(z);
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   FLOATING_DATA_TYPE_SEQ, BOOL_SEQ)
}

}  // namespace test

}  // namespace oneflow
