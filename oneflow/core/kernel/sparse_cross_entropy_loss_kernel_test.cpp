#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/common/switch_func.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename PredType>
struct SparseCrossEntropyLossTestUtil final {
#define CROSS_ENTROPY_LOSS_TEST_UTIL_ENTRY(func_name, T) \
  SparseCrossEntropyLossTestUtil<device_type, PredType>::template func_name<T>
  DEFINE_STATIC_SWITCH_FUNC(
      void, Test, CROSS_ENTROPY_LOSS_TEST_UTIL_ENTRY,
      MAKE_STRINGIZED_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));

  template<typename LabelType>
  static void Test(OpKernelTestCase* test_case, const std::string& job_type,
                   const std::string& fw_or_bw) {
    test_case->set_is_train(job_type == "train");
    test_case->set_is_forward(fw_or_bw == "forward");
    test_case->mut_op_conf()->mutable_sparse_cross_entropy_loss_conf();
    BlobDesc* label_blob_desc = new BlobDesc(
        Shape({2}), GetDataType<LabelType>::value, false, false, 1);
    BlobDesc* blob_desc24 = new BlobDesc(
        Shape({2, 4}), GetDataType<PredType>::value, false, false, 1);
    BlobDesc* blob_desc2 =
        new BlobDesc(Shape({2}), GetDataType<PredType>::value, false, false, 1);
    test_case->InitBlob<PredType>(
        "prediction", blob_desc24,
        {0.0320586041, 0.0871443227, 0.2368828356, 0.6439142823, 0.2500000000,
         0.2500000000, 0.2500000000, 0.2500000000});
    test_case->InitBlob<LabelType>("label", label_blob_desc, {2, 0});
    test_case->ForwardCheckBlob<PredType>(
        "loss", blob_desc2, {1.440189624642414, 1.3862943611198906});
    test_case->BackwardCheckBlob<PredType>(
        GenDiffBn("prediction"), blob_desc24,
        {0, 0, -4.221496240819231, 0, -4.0, 0, 0, 0});
  }
};

template<DeviceType device_type, typename PredType>
void SparseCrossEntropyLossKernelTestCase(OpKernelTestCase* test_case,
                                          const std::string& label_type,
                                          const std::string& job_type,
                                          const std::string& fw_or_bw) {
  SparseCrossEntropyLossTestUtil<device_type, PredType>::SwitchTest(
      SwitchCase(label_type), test_case, job_type, fw_or_bw);
}

TEST_CPU_AND_GPU_OPKERNEL(SparseCrossEntropyLossKernelTestCase,
                          FLOATING_DATA_TYPE_SEQ,
                          OF_PP_SEQ_MAP(OF_PP_PAIR_FIRST, INT_DATA_TYPE_SEQ),
                          (train)(predict), (forward)(backward));

}  // namespace test

}  // namespace oneflow
