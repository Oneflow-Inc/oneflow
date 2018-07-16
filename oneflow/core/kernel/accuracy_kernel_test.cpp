#include "oneflow/core/kernel/opkernel_test_case.h"
#include "oneflow/core/common/switch_func.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename PredType>
struct AccuracyTestUtil final {
#define ACCURACY_TEST_UTIL_ENTRY(func_name, T) \
  AccuracyTestUtil<device_type, PredType>::template func_name<T>
  DEFINE_STATIC_SWITCH_FUNC(void, Test, ACCURACY_TEST_UTIL_ENTRY,
                            MAKE_STRINGIZED_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));

  template<typename LabelType>
  static void Test(OpKernelTestCase* test_case, const std::string& job_type,
                   const std::string& fw_or_bw) {
    test_case->set_is_train(job_type == "train");
    test_case->set_is_forward(fw_or_bw == "forward");
    AccuracyOpConf* accuracy_conf = test_case->mut_op_conf()->mutable_accuracy_conf();
    accuracy_conf->set_prediction("test/prediction");
    accuracy_conf->set_label("test/label");
    accuracy_conf->set_accuracy("test/accuracy");
    accuracy_conf->set_top_k(3);
    BlobDesc* prediction_blob_desc =
        new BlobDesc(Shape({10, 5}), GetDataType<PredType>::value, false, false, 1);
    BlobDesc* label_blob_desc =
        new BlobDesc(Shape({10}), GetDataType<LabelType>::value, false, false, 1);
    BlobDesc* accuracy_blob_desc =
        new BlobDesc(Shape({1}), GetDataType<PredType>::value, false, false, 1);
    test_case->template InitBlob<PredType>(
        "prediction", prediction_blob_desc,
        {4.26386421, 9.95010348, 9.91810292, 0.48375106, 6.64594865, 8.05952355, 6.10698666,
         3.00538932, 0.85184578, 2.07455643, 3.83561549, 3.09892793, 8.03172383, 6.31505591,
         8.27174327, 6.85749046, 9.17082087, 2.75073689, 2.75332767, 9.59847227, 1.73445035,
         0.08238581, 3.83698503, 7.04001947, 6.93058367, 2.21650175, 4.43790294, 1.03987194,
         2.50459141, 4.63530169, 3.91737537, 9.57451706, 6.42044601, 6.69970151, 8.11969361,
         8.47881892, 8.08534761, 0.22607914, 3.28111424, 8.59098739, 1.83841795, 3.76625112,
         9.40150949, 8.43572707, 0.56068475, 8.30401856, 9.45218381, 8.35593787, 0.17762226,
         0.17188453});
    // the first and last label are wrong
    test_case->template InitBlob<LabelType>("label", label_blob_desc,
                                            {3, 1, 3, 4, 3, 4, 1, 4, 1, 4});

    test_case->template ForwardCheckBlob<PredType>("accuracy", accuracy_blob_desc, {0.8});
  }
};

template<DeviceType device_type, typename PredType>
void AccuracyKernelTestCase(OpKernelTestCase* test_case, const std::string& label_type,
                            const std::string& job_type, const std::string& fw_or_bw) {
  AccuracyTestUtil<device_type, PredType>::SwitchTest(SwitchCase(label_type), test_case, job_type,
                                                      fw_or_bw);
}

TEST_CPU_AND_GPU_OPKERNEL(AccuracyKernelTestCase, FLOATING_DATA_TYPE_SEQ,
                          OF_PP_SEQ_MAP(OF_PP_PAIR_FIRST, INT_DATA_TYPE_SEQ), (predict), (forward));

}  // namespace test

}  // namespace oneflow