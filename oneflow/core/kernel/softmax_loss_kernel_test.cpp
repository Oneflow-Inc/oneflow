#include "oneflow/core/kernel/softmax_loss_kernel.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename PredType, typename LabelType>
std::function<Blob*(const std::string&)> BuildBnInOp2Blob() {
  auto bn2blob_ptr = new HashMap<std::string, Blob*>;
  BlobDesc* blob_desc24 =
      new BlobDesc(Shape({2, 4}), GetDataType<PredType>::val, false);
  BlobDesc* blob_desc2 =
      new BlobDesc(Shape({2}), GetDataType<PredType>::val, false);
  BlobDesc* blob_desc1 =
      new BlobDesc(Shape({1}), GetDataType<PredType>::val, false);
  (*bn2blob_ptr)["prediction"] =
      KTCommon<device_type, PredType>::CreateBlobWithSpecifiedVal(
          blob_desc24, {1, 2, 3, 4, 0, 0, 0, 0});
  (*bn2blob_ptr)["label"] =
      KTCommon<device_type, LabelType>::CreateBlobWithSpecifiedVal(
          new BlobDesc(Shape({2}), GetDataType<LabelType>::val, false), {2, 0});
  (*bn2blob_ptr)["prob"] =
      KTCommon<device_type, PredType>::CreateBlobWithRandomVal(blob_desc24);
  (*bn2blob_ptr)["tmp_1D"] =
      KTCommon<device_type, PredType>::CreateBlobWithRandomVal(blob_desc2);
  (*bn2blob_ptr)["loss"] =
      KTCommon<device_type, PredType>::CreateBlobWithRandomVal(blob_desc1);
  (*bn2blob_ptr)["prediction_diff"] =
      KTCommon<device_type, PredType>::CreateBlobWithRandomVal(blob_desc24);
  (*bn2blob_ptr)["expected_loss"] =
      KTCommon<device_type, PredType>::CreateBlobWithSpecifiedVal(blob_desc1,
                                                                  {2.826484f});
  (*bn2blob_ptr)["expected_prediction_diff"] =
      KTCommon<device_type, PredType>::CreateBlobWithSpecifiedVal(
          blob_desc24, {0.0320586f, 0.0871443f, -0.7631172f, 0.6439143f, -0.75f,
                        0.25f, 0.25f, 0.25f});
  return [bn2blob_ptr](const std::string& bn) { return bn2blob_ptr->at(bn); };
}

template<DeviceType device_type, typename PredType, typename LabelType>
Kernel* BuildSoftmaxLossKernel() {
  OperatorConf op_conf;
  op_conf.set_name("softmax_loss_op_test");
  SoftmaxLossOpConf* softmax_loss_conf = op_conf.mutable_softmax_loss_conf();
  softmax_loss_conf->mutable_prediction()->set_name("softmax_loss/prediction");
  softmax_loss_conf->mutable_prediction()->set_data_type(
      GetDataType<PredType>::val);
  softmax_loss_conf->mutable_label()->set_name("softmax_loss/label");
  softmax_loss_conf->mutable_label()->set_data_type(
      GetDataType<LabelType>::val);
  softmax_loss_conf->mutable_loss()->set_name("softmax_loss/loss");
  softmax_loss_conf->mutable_loss()->set_data_type(GetDataType<PredType>::val);
  auto softmax_loss_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  softmax_loss_op->ToProto(&op_proto);
  auto softmax_loss_kernel =
      new SoftmaxLossKernel<device_type, PredType, LabelType>();
  softmax_loss_kernel->InitFromOpProto(op_proto);
  return softmax_loss_kernel;
}

template<DeviceType device_type, typename PredType, typename LabelType>
void TestSoftmaxLossKernel() {
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);
  auto BnInOp2Blob = BuildBnInOp2Blob<device_type, PredType, LabelType>();
  auto softmax_loss_kernel =
      BuildSoftmaxLossKernel<device_type, PredType, LabelType>();
  softmax_loss_kernel->Forward(ctx, BnInOp2Blob);
  SyncStream<device_type>(&ctx);
  KTCommon<device_type, PredType>::CheckResult(BnInOp2Blob, "loss",
                                               "expected_loss");
  KTCommon<device_type, PredType>::CheckResult(BnInOp2Blob, "prediction_diff",
                                               "expected_prediction_diff");
}

}  // namespace

}  // namespace test

TEST(SoftmaxLossKernel, softmax_loss_kernel_fw_and_bp) {
#define MAKE_ENTRY(device_type, pred_type_pair, label_type_pair)             \
  test::TestSoftmaxLossKernel<device_type, OF_PP_PAIR_FIRST(pred_type_pair), \
                              OF_PP_PAIR_FIRST(label_type_pair)>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)
#undef MAKE_ENTRY
}

}  // namespace oneflow
