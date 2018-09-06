#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/common/switch_func.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename PredType>
struct HingeLossTestUtil final {
#define HINGE_LOSS_TEST_UTIL_ENTRY(func_name, T) \
  HingeLossTestUtil<device_type, PredType>::template func_name<T>

  DEFINE_STATIC_SWITCH_FUNC(void, Test, HINGE_LOSS_TEST_UTIL_ENTRY,
                            MAKE_STRINGIZED_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));

  template<typename LabelType>
  static void Test(OpKernelTestCase* test_case, const std::string& job_type,
                   const std::string& fw_or_bw) {
    test_case->set_is_train(job_type == "train");
    test_case->set_is_forward(fw_or_bw == "forward");
    HingeLossOpConf* hinge_loss_conf = test_case->mut_op_conf()->mutable_hinge_loss_conf();
    hinge_loss_conf->set_norm(Norm::L2);
    hinge_loss_conf->set_label("test/label");
    hinge_loss_conf->set_prediction("test/prediction");
    hinge_loss_conf->set_loss("test/loss");
    BlobDesc* label_blob_desc =
        new BlobDesc(Shape({2}), GetDataType<LabelType>::value, false, false, 1);
    BlobDesc* pred_blob_desc =
        new BlobDesc(Shape({2, 5}), GetDataType<PredType>::value, false, false, 1);
    BlobDesc* loss_blob_desc =
        new BlobDesc(Shape({2}), GetDataType<PredType>::value, false, false, 1);
    test_case->InitBlob<LabelType>("label", label_blob_desc, {2, 2});
    test_case->InitBlob<PredType>(
        "prediction", pred_blob_desc,
        {-1.73, -1.24, 0.89, -0.99, 0.05, -1.73, -1.24, 0.89, -0.99, 0.05});
    test_case->ForwardCheckBlob<PredType>("loss", loss_blob_desc, {1.1147, 1.1147});
    test_case->BackwardCheckBlob<PredType>(
        GenDiffBn("prediction"), pred_blob_desc,
        {0.00, 0.00, -0.22, 0.02, 2.10, 0.00, 0.00, -0.22, 0.02, 2.10});
  }
};

template<DeviceType device_type, typename PredType>
void HingeLossKernelTestCase(OpKernelTestCase* test_case, const std::string& label_type,
                             const std::string& job_type, const std::string& fw_or_bw) {
  HingeLossTestUtil<device_type, PredType>::SwitchTest(SwitchCase(label_type), test_case, job_type,
                                                       fw_or_bw);
}

TEST_CPU_AND_GPU_OPKERNEL(HingeLossKernelTestCase, FLOATING_DATA_TYPE_SEQ,
                          OF_PP_SEQ_MAP(OF_PP_PAIR_FIRST, INT_DATA_TYPE_SEQ), (train)(predict),
                          (forward)(backward));

}  // namespace test

}  // namespace oneflow
