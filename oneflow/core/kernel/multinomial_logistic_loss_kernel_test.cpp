#include "oneflow/core/kernel/multinomial_logistic_loss_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename FloatingPointType>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobPtr() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  auto bn2blob_ptr = new HashMap<std::string, Blob*>;

  FloatingPointType prediction_mat[40] = {
      1.314913e-05, 4.500352e-05, 3.436922e-06, 9.997171e-01, 1.758189e-08,
      8.714826e-06, 1.898432e-08, 1.979446e-07, 2.107414e-04, 1.660027e-06,
      7.234089e-06, 2.464892e-06, 4.696337e-07, 9.974370e-01, 6.605830e-06,
      2.160112e-03, 2.135477e-07, 2.358040e-04, 1.226367e-04, 2.739524e-05,
      1.980253e-06, 2.231704e-05, 9.996368e-01, 2.880224e-04, 4.716213e-06,
      3.297473e-07, 8.703138e-08, 1.498134e-06, 4.409142e-05, 1.453549e-07,
      5.988552e-04, 3.569204e-05, 4.662264e-04, 2.320096e-04, 6.499564e-02,
      6.946083e-04, 6.789275e-04, 7.571414e-03, 6.086799e-03, 9.186398e-01};
  (*bn2blob_ptr)["prediction"] =
      KTCommon::CreateBlobWithVector({4, 10}, prediction_mat);

  FloatingPointType loss_mat[1] = {0};
  (*bn2blob_ptr)["loss"] = KTCommon::CreateBlobWithVector({1, 1}, loss_mat);

  FloatingPointType loss_buff_mat[4] = {0};
  (*bn2blob_ptr)["loss_buffer"] =
      KTCommon::CreateBlobWithVector({1, 1}, loss_buff_mat);

  FloatingPointType expected_loss_mat[1] = {2.201842e-02};
  (*bn2blob_ptr)["expected_loss"] =
      KTCommon::CreateBlobWithVector({1, 1}, expected_loss_mat);

  // label is not one hot
  FloatingPointType label_mat[4] = {3, 3, 2, 9};
  (*bn2blob_ptr)["label"] = KTCommon::CreateBlobWithVector({1, 4}, label_mat);

  FloatingPointType prediction_diff_mat[40] = {0};
  for (int64_t i = 0; i < 40; ++i) {
    prediction_diff_mat[i] =
        (FloatingPointType)rand() / (FloatingPointType)RAND_MAX;
  }
  (*bn2blob_ptr)["prediction_diff"] =
      KTCommon::CreateBlobWithVector({4, 10}, prediction_diff_mat);
  FloatingPointType expected_prediction_diff_mat[40] = {
      -0.000000e+00, -0.000000e+00, -0.000000e+00, -2.500708e-01,
      -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00,
      -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00,
      -0.000000e+00, -2.506424e-01, -0.000000e+00, -0.000000e+00,
      -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00,
      -0.000000e+00, -0.000000e+00, -2.500908e-01, -0.000000e+00,
      -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00,
      -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00,
      -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00,
      -0.000000e+00, -0.000000e+00, -0.000000e+00, -2.721415e-01};
  (*bn2blob_ptr)["expected_prediction_diff"] =
      KTCommon::CreateBlobWithVector({4, 10}, expected_prediction_diff_mat);

  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename FloatingPointType>
Kernel* BuildMultinomialLogisticLossKernel() {
  OperatorConf op_conf;
  op_conf.set_name("multinomial_logistic_loss_test");
  MultinomialLogisticLossOpConf* multinomial_logistic_loss_conf =
      op_conf.mutable_multinomial_logistic_loss_conf();
  multinomial_logistic_loss_conf->set_prediction("mll/prediction");
  multinomial_logistic_loss_conf->set_label("mll/label");
  multinomial_logistic_loss_conf->set_loss("mll/loss");
  auto multinomial_logistic_loss_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  multinomial_logistic_loss_op->ToProto(&op_proto);
  auto multinomial_logistic_loss_kernel =
      new MultinomialLogisticLossKernel<device_type, FloatingPointType>();
  multinomial_logistic_loss_kernel->InitFromOpProto(op_proto);
  return multinomial_logistic_loss_kernel;
}

template<DeviceType device_type, typename FloatingPointType>
void TestMultinomialLogisticLossKernel() {
  using KTCommon = KernelTestCommon<device_type, FloatingPointType>;
  KernelCtx ctx;
  KTCommon::BuildKernelCtx(&ctx);
  auto BnInOp2BlobPtr = BuildBnInOp2BlobPtr<device_type, FloatingPointType>();
  auto multinomial_logistic_loss_kernel =
      BuildMultinomialLogisticLossKernel<device_type, FloatingPointType>();
  multinomial_logistic_loss_kernel->Forward(ctx, BnInOp2BlobPtr);
  KTCommon::SyncStream(&ctx);
  KTCommon::CheckResult(BnInOp2BlobPtr, "loss", "expected_loss");
  KTCommon::CheckResult(BnInOp2BlobPtr, "prediction_diff",
                        "expected_prediction_diff");
}

}  // namespace

}  // namespace test

TEST(MultinomialLogisticLossKernel, multinomial_logistic_loss_kernel_cpu) {
  test::TestMultinomialLogisticLossKernel<DeviceType::kCPU, float>();
  test::TestMultinomialLogisticLossKernel<DeviceType::kCPU, double>();
}

TEST(MultinomialLogisticLossKernel, multinomial_logistic_loss_kernel_gpu) {
  test::TestMultinomialLogisticLossKernel<DeviceType::kGPU, float>();
  test::TestMultinomialLogisticLossKernel<DeviceType::kGPU, double>();
}

}  // namespace oneflow
