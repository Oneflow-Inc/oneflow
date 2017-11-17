#include "oneflow/core/kernel/multinomial_logistic_loss_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename PredType, typename LabelType>
std::function<Blob*(const std::string&)> BuildBnInOp2Blob() {
  auto bn2blob_ptr = new HashMap<std::string, Blob*>;

  using KTC_PRED = KTCommon<device_type, PredType>;
  using KTC_LABEL = KTCommon<device_type, LabelType>;
  (*bn2blob_ptr)["prediction"] = KTC_PRED::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({4, 10}), GetDataType<PredType>::val, false),
      {1.314913e-05f, 4.500352e-05f, 3.436922e-06f, 9.997171e-01f,
       1.758189e-08f, 8.714826e-06f, 1.898432e-08f, 1.979446e-07f,
       2.107414e-04f, 1.660027e-06f, 7.234089e-06f, 2.464892e-06f,
       4.696337e-07f, 9.974370e-01f, 6.605830e-06f, 2.160112e-03f,
       2.135477e-07f, 2.358040e-04f, 1.226367e-04f, 2.739524e-05f,
       1.980253e-06f, 2.231704e-05f, 9.996368e-01f, 2.880224e-04f,
       4.716213e-06f, 3.297473e-07f, 8.703138e-08f, 1.498134e-06f,
       4.409142e-05f, 1.453549e-07f, 5.988552e-04f, 3.569204e-05f,
       4.662264e-04f, 2.320096e-04f, 6.499564e-02f, 6.946083e-04f,
       6.789275e-04f, 7.571414e-03f, 6.086799e-03f, 9.186398e-01f});

  (*bn2blob_ptr)["loss"] = KTC_PRED::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 1}), GetDataType<PredType>::val, false), {0.0});

  (*bn2blob_ptr)["loss_buffer"] = KTC_PRED::CreateBlobWithRandomVal(
      new BlobDesc(Shape({4, 1}), GetDataType<PredType>::val, false));

  (*bn2blob_ptr)["expected_loss"] = KTC_PRED::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 1}), GetDataType<PredType>::val, false),
      {8.807367e-02f});

  (*bn2blob_ptr)["label"] = KTC_LABEL::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 4}), GetDataType<LabelType>::val, false),
      {3, 3, 2, 9});

  (*bn2blob_ptr)["prediction_diff"] = KTC_PRED::CreateBlobWithRandomVal(
      new BlobDesc(Shape({4, 10}), GetDataType<PredType>::val, false));

  (*bn2blob_ptr)["expected_prediction_diff"] =
      KTC_PRED::CreateBlobWithSpecifiedVal(
          new BlobDesc(Shape({4, 10}), GetDataType<PredType>::val, false),
          {-0.000000e+00f, -0.000000e+00f, -0.000000e+00f, -1.000283e+00f,
           -0.000000e+00f, -0.000000e+00f, -0.000000e+00f, -0.000000e+00f,
           -0.000000e+00f, -0.000000e+00f, -0.000000e+00f, -0.000000e+00f,
           -0.000000e+00f, -1.002570e+00f, -0.000000e+00f, -0.000000e+00f,
           -0.000000e+00f, -0.000000e+00f, -0.000000e+00f, -0.000000e+00f,
           -0.000000e+00f, -0.000000e+00f, -1.000363e+00f, -0.000000e+00f,
           -0.000000e+00f, -0.000000e+00f, -0.000000e+00f, -0.000000e+00f,
           -0.000000e+00f, -0.000000e+00f, -0.000000e+00f, -0.000000e+00f,
           -0.000000e+00f, -0.000000e+00f, -0.000000e+00f, -0.000000e+00f,
           -0.000000e+00f, -0.000000e+00f, -0.000000e+00f, -1.088566e+00f});

  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename PredType, typename LabelType>
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
      new MultinomialLogisticLossKernel<device_type, PredType, LabelType>();
  multinomial_logistic_loss_kernel->InitFromOpProto(op_proto);
  return multinomial_logistic_loss_kernel;
}

template<DeviceType device_type, typename PredType, typename LabelType>
void TestMultinomialLogisticLossKernel() {
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);
  auto BnInOp2Blob = BuildBnInOp2Blob<device_type, PredType, LabelType>();
  auto multinomial_logistic_loss_kernel =
      BuildMultinomialLogisticLossKernel<device_type, PredType, LabelType>();
  multinomial_logistic_loss_kernel->Forward(ctx, BnInOp2Blob);
  SyncStream<device_type>(&ctx);
  KTCommon<device_type, PredType>::CheckResult(BnInOp2Blob, "loss",
                                               "expected_loss");
  KTCommon<device_type, PredType>::CheckResult(BnInOp2Blob, "prediction_diff",
                                               "expected_prediction_diff");
}

}  // namespace

}  // namespace test

TEST(MultinomialLogisticLossKernel, multinomial_logistic_loss_kernel) {
#define MAKE_ENTRY(device_type, pred_type_pair, label_type_pair) \
  test::TestMultinomialLogisticLossKernel<                       \
      device_type, OF_PP_PAIR_FIRST(pred_type_pair),             \
      OF_PP_PAIR_FIRST(label_type_pair)>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)
}

}  // namespace oneflow
