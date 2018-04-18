#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

DiffKernelImplTestCase* DiffNormalizationKernelImpl(const std::string& job_type,
                                                    const std::string& fw_or_bw,
                                                    const std::string& cpp_type,
                                                    const std::string& scale,
                                                    const std::string& center,
                                                    const std::string& axis) {
  auto* test_case =
      new DiffKernelImplTestCase(job_type == "train", fw_or_bw == "forward",
                                 DataType4CppTypeString(cpp_type));
  auto* conf = test_case->mut_op_conf()->mutable_normalization_conf();
  bool use_scale = (scale == "scale");
  bool use_center = (center == "center");
  conf->set_scale(use_scale);
  conf->set_center(use_center);
  conf->set_axis(std::stoi(axis));
  conf->set_use_first_piece_init_moving(true);
  test_case->set_initiate_kernel_ctx(
      [=](const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
        std::tuple<int64_t, std::function<const Blob*(const std::string&)>>*
            other_val =
                new std::tuple<int64_t,
                               std::function<const Blob*(const std::string&)>>(
                    0, [&](const std::string& lbn) -> const Blob* {
                      if (lbn.find("mean") != std::string::npos) {
                        return BnInOp2Blob("moving_mean");
                      } else {
                        return BnInOp2Blob("moving_variance");
                      }
                    });
        test_case->mut_kernel_ctx()->other = other_val;
      });
  Shape shape({10, 10, 28, 28});
  std::list<std::string> input_or_weight_bns{"in"};
  std::list<std::string> input_diff_or_weight_diff_bns{GenDiffBn("in")};
  if (use_scale) {
    input_or_weight_bns.push_back("gamma");
    input_diff_or_weight_diff_bns.push_back(GenDiffBn("gamma"));
  }
  if (use_center) {
    input_or_weight_bns.push_back("beta");
    input_diff_or_weight_diff_bns.push_back(GenDiffBn("beta"));
  }
  test_case->SetBlobNames(input_or_weight_bns, {"out"}, {GenDiffBn("out")},
                          input_diff_or_weight_diff_bns);
  test_case->SetInputBlobDesc("in", shape, DataType4CppTypeString(cpp_type));
  test_case->SetInputBlobDesc(GenDiffBn("out"), shape,
                              DataType4CppTypeString(cpp_type));
  return test_case;
}
TEST_DIFF_KERNEL_IMPL(DiffNormalizationKernelImpl, (train)(predict),
                      (forward)(backward),
                      OF_PP_SEQ_MAP(OF_PP_PAIR_FIRST, FLOATING_DATA_TYPE_SEQ),
                      (scale)(no_scale), (center)(no_center), (0)(1));

template<DeviceType device_type, typename T>
void NormalizationTestCase_single_number(OpKernelTestCase* norm_test_case,
                                         const std::string& job_type,
                                         const std::string& fw_or_bw) {
  norm_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  bool is_train = (job_type == "train");
  bool is_forward = (fw_or_bw == "forward");
  norm_test_case->set_is_train(is_train);
  norm_test_case->set_is_forward(is_forward);
  auto* conf = norm_test_case->mut_op_conf()->mutable_normalization_conf();
  bool scale = true;
  bool center = true;
  conf->set_scale(scale);
  conf->set_center(center);

  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  CHECK(one_blob_desc->shape().elem_cnt() == 1);
  norm_test_case->template InitBlob<T>("in", one_blob_desc, {1});

  Blob* mmean_blob =
      norm_test_case->template InitBlob<T>("moving_mean", one_blob_desc, {0});
  Blob* mvariance_blob = norm_test_case->template InitBlob<T>(
      "moving_variance", one_blob_desc, {1});

  if (center) {
    norm_test_case->template InitBlob<T>("beta", one_blob_desc, {0});
  }
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {1});
  }
  norm_test_case->template InitBlob<T>(GenDiffBn("out"), one_blob_desc, {1});

  std::tuple<int64_t,
             std::function<const Blob*(const std::string&)>>* other_val =
      new std::tuple<int64_t, std::function<const Blob*(const std::string&)>>(
          1, [=](const std::string& lbn) -> const Blob* {
            if (lbn.find("mean") != std::string::npos) {
              return mmean_blob;
            } else {
              return mvariance_blob;
            }
          });
  norm_test_case->mut_kernel_ctx()->other = other_val;

  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_mean", one_blob_desc,
                                                 {1});
  }
  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_variance", one_blob_desc,
                                                 {0});
  }
  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("inv_var", one_blob_desc,
                                                 {31.622776});
    norm_test_case->template ForwardCheckBlob<T>("out", one_blob_desc, {0});
    norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                                 {0.01}, false);
    norm_test_case->template ForwardCheckBlob<T>("moving_variance",
                                                 one_blob_desc, {0.99}, false);
  } else {
    norm_test_case->template ForwardCheckBlob<T>("inv_var", one_blob_desc,
                                                 {0.9995003});
    norm_test_case->template ForwardCheckBlob<T>("out", one_blob_desc,
                                                 {0.9995003});
    norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                                 {0}, false);
    norm_test_case->template ForwardCheckBlob<T>("moving_variance",
                                                 one_blob_desc, {1}, false);
  }

  norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("in"), one_blob_desc,
                                                {0});
  if (center) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("beta"),
                                                  one_blob_desc, {1});
  }
  if (scale) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("gamma"),
                                                  one_blob_desc, {0});
  }
}

TEST_CPU_AND_GPU_OPKERNEL(NormalizationTestCase_single_number,
                          FLOATING_DATA_TYPE_SEQ, (train)(predict),
                          (forward)(backward));

template<DeviceType device_type, typename T>
void NormalizationTestCase_first_piece(OpKernelTestCase* norm_test_case,
                                       const std::string& job_type,
                                       const std::string& fw_or_bw) {
  norm_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  bool is_train = (job_type == "train");
  norm_test_case->set_is_train(is_train);
  norm_test_case->set_is_forward(fw_or_bw == "forward");
  auto* conf = norm_test_case->mut_op_conf()->mutable_normalization_conf();
  bool scale = true;
  bool center = true;
  conf->set_scale(scale);
  conf->set_center(center);
  conf->set_axis(0);
  conf->set_use_first_piece_init_moving(true);

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 5}), GetDataType<T>::value, false, false, 1);
  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("in", blob_desc, {1, 2, 3, 4, 5});

  Blob* mmean_blob;
  Blob* mvariance_blob;
  if (is_train) {
    mmean_blob =
        norm_test_case->template InitBlob<T>("moving_mean", one_blob_desc, {0});
    mvariance_blob = norm_test_case->template InitBlob<T>("moving_variance",
                                                          one_blob_desc, {1});
  } else {
    mmean_blob = norm_test_case->template InitBlob<T>("moving_mean",
                                                      one_blob_desc, {3.0});
    mvariance_blob = norm_test_case->template InitBlob<T>("moving_variance",
                                                          one_blob_desc, {2.0});
  }

  if (center) norm_test_case->template InitBlob<T>("beta", one_blob_desc, {0});
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {10});
  }
  norm_test_case->template InitBlob<T>(GenDiffBn("out"), blob_desc,
                                       {1, 2, 2, 6, 3});

  std::tuple<int64_t,
             std::function<const Blob*(const std::string&)>>* other_val =
      new std::tuple<int64_t, std::function<const Blob*(const std::string&)>>(
          0, [=](const std::string& lbn) -> const Blob* {
            if (lbn.find("mean") != std::string::npos) {
              return mmean_blob;
            } else {
              return mvariance_blob;
            }
          });
  norm_test_case->mut_kernel_ctx()->other = other_val;

  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_mean", one_blob_desc,
                                                 {3.0});
  }
  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_variance", one_blob_desc,
                                                 {2.0});
  }
  norm_test_case->template ForwardCheckBlob<T>("inv_var", one_blob_desc,
                                               {0.706930});
  norm_test_case->template ForwardCheckBlob<T>(
      "out", blob_desc, {-14.13860, -7.06930, 0, 7.06930, 14.13860});
  norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                               {3.0}, false);
  norm_test_case->template ForwardCheckBlob<T>("moving_variance", one_blob_desc,
                                               {2.0}, false);

  norm_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), blob_desc,
      {-1.419513, -0.002826, -5.655441, 16.969148, -9.891368});
  if (center) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("beta"),
                                                  one_blob_desc, {14});
  }
  if (scale) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("gamma"),
                                                  one_blob_desc, {5.655440});
  }
}

TEST_CPU_AND_GPU_OPKERNEL(NormalizationTestCase_first_piece,
                          FLOATING_DATA_TYPE_SEQ, (train)(predict),
                          (forward)(backward));

template<DeviceType device_type, typename T>
void NormalizationTestCase_second_piece_train_different_preblob(
    OpKernelTestCase* norm_test_case, const std::string& job_type,
    const std::string& fw_or_bw) {
  norm_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  bool is_train = (job_type == "train");
  norm_test_case->set_is_train(is_train);
  norm_test_case->set_is_forward(fw_or_bw == "forward");
  auto* conf = norm_test_case->mut_op_conf()->mutable_normalization_conf();
  bool scale = true;
  bool center = true;
  conf->set_scale(scale);
  conf->set_center(center);
  conf->set_axis(0);

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 5}), GetDataType<T>::value, false, false, 1);
  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("in", blob_desc, {1, 2, 3, 4, 5});

  norm_test_case->template InitBlob<T>("moving_mean", one_blob_desc, {11.0});
  norm_test_case->template InitBlob<T>("moving_variance", one_blob_desc,
                                       {20.0});

  Blob* pre_mean_blob = norm_test_case->template InitBlob<T>(
      "pre_moving_mean", one_blob_desc, {3.0});
  Blob* pre_variance_blob = norm_test_case->template InitBlob<T>(
      "pre_moving_variance", one_blob_desc, {2.0});

  if (center) norm_test_case->template InitBlob<T>("beta", one_blob_desc, {0});
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {10});
  }
  norm_test_case->template InitBlob<T>(GenDiffBn("out"), blob_desc,
                                       {1, 2, 2, 6, 3});

  std::tuple<int64_t,
             std::function<const Blob*(const std::string&)>>* other_val =
      new std::tuple<int64_t, std::function<const Blob*(const std::string&)>>(
          1, [=](const std::string& lbn) -> const Blob* {
            if (lbn.find("mean") != std::string::npos) {
              return pre_mean_blob;
            } else {
              return pre_variance_blob;
            }
          });
  norm_test_case->mut_kernel_ctx()->other = other_val;

  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_mean", one_blob_desc,
                                                 {3.0});
  }
  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_variance", one_blob_desc,
                                                 {2.0});
  }
  norm_test_case->template ForwardCheckBlob<T>("inv_var", one_blob_desc,
                                               {0.706930});
  norm_test_case->template ForwardCheckBlob<T>(
      "out", blob_desc, {-14.13860, -7.06930, 0, 7.06930, 14.13860});
  norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                               {3.0}, false);
  norm_test_case->template ForwardCheckBlob<T>("moving_variance", one_blob_desc,
                                               {2.0}, false);

  norm_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), blob_desc,
      {-1.419513, -0.002826, -5.655441, 16.969148, -9.891368});
  if (center) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("beta"),
                                                  one_blob_desc, {14});
  }
  if (scale) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("gamma"),
                                                  one_blob_desc, {5.655440});
  }
}

TEST_CPU_AND_GPU_OPKERNEL(
    NormalizationTestCase_second_piece_train_different_preblob,
    FLOATING_DATA_TYPE_SEQ, (train), (forward)(backward));

template<DeviceType device_type, typename T>
void NormalizationTestCase_second_piece_without_gamma(
    OpKernelTestCase* norm_test_case, const std::string& job_type,
    const std::string& fw_or_bw) {
  norm_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  bool is_train = (job_type == "train");
  norm_test_case->set_is_train(is_train);
  norm_test_case->set_is_forward(fw_or_bw == "forward");
  auto* conf = norm_test_case->mut_op_conf()->mutable_normalization_conf();
  bool scale = false;
  bool center = true;
  conf->set_scale(scale);
  conf->set_center(center);
  conf->set_axis(0);

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 5}), GetDataType<T>::value, false, false, 1);
  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("in", blob_desc, {1, 2, 3, 4, 5});

  Blob* mmean_blob =
      norm_test_case->template InitBlob<T>("moving_mean", one_blob_desc, {3.0});
  Blob* mvariance_blob = norm_test_case->template InitBlob<T>(
      "moving_variance", one_blob_desc, {2.0});

  if (center) norm_test_case->template InitBlob<T>("beta", one_blob_desc, {1});
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {10});
  }
  norm_test_case->template InitBlob<T>(GenDiffBn("out"), blob_desc,
                                       {1, 2, 2, 6, 3});

  std::tuple<int64_t,
             std::function<const Blob*(const std::string&)>>* other_val =
      new std::tuple<int64_t, std::function<const Blob*(const std::string&)>>(
          1, [=](const std::string& lbn) -> const Blob* {
            if (lbn.find("mean") != std::string::npos) {
              return mmean_blob;
            } else {
              return mvariance_blob;
            }
          });
  norm_test_case->mut_kernel_ctx()->other = other_val;

  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_mean", one_blob_desc,
                                                 {3.0});
  }
  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_variance", one_blob_desc,
                                                 {2.0});
  }
  norm_test_case->template ForwardCheckBlob<T>("inv_var", one_blob_desc,
                                               {0.706930});
  norm_test_case->template ForwardCheckBlob<T>(
      "out", blob_desc, {1 - 1.413860, 1 - 0.706930, 1, 1.706930, 2.413860});
  norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                               {3.0}, false);
  norm_test_case->template ForwardCheckBlob<T>("moving_variance", one_blob_desc,
                                               {2.0}, false);

  norm_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), blob_desc,
      {-0.141951, -0.000283, -0.565544, 1.696915, -0.989137});
  if (center) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("beta"),
                                                  one_blob_desc, {14});
  }
  if (scale) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("gamma"),
                                                  one_blob_desc, {5.655440});
  }
}

TEST_CPU_AND_GPU_OPKERNEL(NormalizationTestCase_second_piece_without_gamma,
                          FLOATING_DATA_TYPE_SEQ, (train)(predict),
                          (forward)(backward));

template<DeviceType device_type, typename T>
void NormalizationTestCase_second_piece_without_beta(
    OpKernelTestCase* norm_test_case, const std::string& job_type,
    const std::string& fw_or_bw) {
  norm_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  bool is_train = (job_type == "train");
  norm_test_case->set_is_train(is_train);
  norm_test_case->set_is_forward(fw_or_bw == "forward");
  auto* conf = norm_test_case->mut_op_conf()->mutable_normalization_conf();
  bool scale = true;
  bool center = false;
  conf->set_scale(scale);
  conf->set_center(center);
  conf->set_axis(0);

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 5}), GetDataType<T>::value, false, false, 1);
  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("in", blob_desc, {1, 2, 3, 4, 5});

  Blob* mmean_blob =
      norm_test_case->template InitBlob<T>("moving_mean", one_blob_desc, {3.0});
  Blob* mvariance_blob = norm_test_case->template InitBlob<T>(
      "moving_variance", one_blob_desc, {2.0});

  if (center) norm_test_case->template InitBlob<T>("beta", one_blob_desc, {1});
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {10});
  }
  norm_test_case->template InitBlob<T>(GenDiffBn("out"), blob_desc,
                                       {1, 2, 2, 6, 3});

  std::tuple<int64_t,
             std::function<const Blob*(const std::string&)>>* other_val =
      new std::tuple<int64_t, std::function<const Blob*(const std::string&)>>(
          1, [=](const std::string& lbn) -> const Blob* {
            if (lbn.find("mean") != std::string::npos) {
              return mmean_blob;
            } else {
              return mvariance_blob;
            }
          });
  norm_test_case->mut_kernel_ctx()->other = other_val;

  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_mean", one_blob_desc,
                                                 {3.0});
  }
  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_variance", one_blob_desc,
                                                 {2.0});
  }
  norm_test_case->template ForwardCheckBlob<T>("inv_var", one_blob_desc,
                                               {0.706930});
  norm_test_case->template ForwardCheckBlob<T>(
      "out", blob_desc, {-14.13860, -7.06930, 0, 7.06930, 14.13860});
  norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                               {3.0}, false);
  norm_test_case->template ForwardCheckBlob<T>("moving_variance", one_blob_desc,
                                               {2.0}, false);

  norm_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), blob_desc,
      {-1.419513, -0.002826, -5.655441, 16.969148, -9.891368});
  if (center) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("beta"),
                                                  one_blob_desc, {14});
  }
  if (scale) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("gamma"),
                                                  one_blob_desc, {5.655440});
  }
}

TEST_CPU_AND_GPU_OPKERNEL(NormalizationTestCase_second_piece_without_beta,
                          FLOATING_DATA_TYPE_SEQ, (train)(predict),
                          (forward)(backward));

template<DeviceType device_type, typename T>
void NormalizationTestCase_second_piece_without_beta_and_gamma(
    OpKernelTestCase* norm_test_case, const std::string& job_type,
    const std::string& fw_or_bw) {
  norm_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  bool is_train = (job_type == "train");
  norm_test_case->set_is_train(is_train);
  norm_test_case->set_is_forward(fw_or_bw == "forward");
  auto* conf = norm_test_case->mut_op_conf()->mutable_normalization_conf();
  bool scale = false;
  bool center = false;
  conf->set_scale(scale);
  conf->set_center(center);
  conf->set_axis(0);

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 5}), GetDataType<T>::value, false, false, 1);
  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("in", blob_desc, {1, 2, 3, 4, 5});

  Blob* mmean_blob =
      norm_test_case->template InitBlob<T>("moving_mean", one_blob_desc, {3.0});
  Blob* mvariance_blob = norm_test_case->template InitBlob<T>(
      "moving_variance", one_blob_desc, {2.0});

  if (center) norm_test_case->template InitBlob<T>("beta", one_blob_desc, {1});
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {10});
  }
  norm_test_case->template InitBlob<T>(GenDiffBn("out"), blob_desc,
                                       {1, 2, 2, 6, 3});

  std::tuple<int64_t,
             std::function<const Blob*(const std::string&)>>* other_val =
      new std::tuple<int64_t, std::function<const Blob*(const std::string&)>>(
          1, [=](const std::string& lbn) -> const Blob* {
            if (lbn.find("mean") != std::string::npos) {
              return mmean_blob;
            } else {
              return mvariance_blob;
            }
          });
  norm_test_case->mut_kernel_ctx()->other = other_val;

  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_mean", one_blob_desc,
                                                 {3.0});
  }
  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_variance", one_blob_desc,
                                                 {2.0});
  }
  norm_test_case->template ForwardCheckBlob<T>("inv_var", one_blob_desc,
                                               {0.706930});
  norm_test_case->template ForwardCheckBlob<T>(
      "out", blob_desc, {-1.413860, -0.706930, 0, 0.706930, 1.413860});
  norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                               {3.0}, false);
  norm_test_case->template ForwardCheckBlob<T>("moving_variance", one_blob_desc,
                                               {2.0}, false);

  norm_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), blob_desc,
      {-0.141951, -0.000283, -0.565544, 1.696915, -0.989137});
  if (center) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("beta"),
                                                  one_blob_desc, {14});
  }
  if (scale) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("gamma"),
                                                  one_blob_desc, {5.655440});
  }
}

TEST_CPU_AND_GPU_OPKERNEL(
    NormalizationTestCase_second_piece_without_beta_and_gamma,
    FLOATING_DATA_TYPE_SEQ, (train)(predict), (forward)(backward));

template<DeviceType device_type, typename T>
void NormalizationTestCase_second_piece_transpose(
    OpKernelTestCase* norm_test_case, const std::string& job_type,
    const std::string& fw_or_bw) {
  norm_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  bool is_train = (job_type == "train");
  norm_test_case->set_is_train(is_train);
  norm_test_case->set_is_forward(fw_or_bw == "forward");
  auto* conf = norm_test_case->mut_op_conf()->mutable_normalization_conf();
  bool scale = true;
  bool center = true;
  conf->set_scale(scale);
  conf->set_center(center);
  conf->set_axis(1);

  BlobDesc* blob_desc =
      new BlobDesc(Shape({5, 2}), GetDataType<T>::value, false, false, 1);
  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({2}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("in", blob_desc,
                                       {1, 2, 2, 12, 3, 0.5, 4, 8, 5, -7.05});
  Blob* mmean_blob;
  Blob* mvariance_blob;
  if (is_train) {
    mmean_blob = norm_test_case->template InitBlob<T>("moving_mean",
                                                      one_blob_desc, {3.0, 0});
    mvariance_blob = norm_test_case->template InitBlob<T>(
        "moving_variance", one_blob_desc, {2.0, 1.0});
  } else {
    mmean_blob = norm_test_case->template InitBlob<T>(
        "moving_mean", one_blob_desc, {3.0, 3.09});
    mvariance_blob = norm_test_case->template InitBlob<T>(
        "moving_variance", one_blob_desc, {2.0, 42.8424});
  }

  if (center)
    norm_test_case->template InitBlob<T>("beta", one_blob_desc, {1, -8});
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {10, 15});
  }
  norm_test_case->template InitBlob<T>(
      GenDiffBn("out"), blob_desc, {1, 0.1, 2, -2, 2, -2.2, 6, 6.6666, 3, 39});

  std::tuple<int64_t,
             std::function<const Blob*(const std::string&)>>* other_val =
      new std::tuple<int64_t, std::function<const Blob*(const std::string&)>>(
          1, [=](const std::string& lbn) -> const Blob* {
            if (lbn.find("mean") != std::string::npos) {
              return mmean_blob;
            } else {
              return mvariance_blob;
            }
          });
  norm_test_case->mut_kernel_ctx()->other = other_val;

  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_mean", one_blob_desc,
                                                 {3.0, 3.090000});
  }
  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("new_variance", one_blob_desc,
                                                 {2.0, 42.842400});
  }
  norm_test_case->template ForwardCheckBlob<T>("inv_var", one_blob_desc,
                                               {0.706930, 0.152777});
  norm_test_case->template ForwardCheckBlob<T>(
      "out", blob_desc,
      {-13.138601, -10.497904, -6.069301, 12.418649, 1.000000, -13.935387,
       8.069301, 3.252028, 15.138601, -31.237385});
  if (is_train) {
    norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                                 {3.0, 0.0309}, false);
    norm_test_case->template ForwardCheckBlob<T>(
        "moving_variance", one_blob_desc, {2.0, 1.418424}, false);
  } else {
    norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                                 {3.0, 3.09}, false);
    norm_test_case->template ForwardCheckBlob<T>(
        "moving_variance", one_blob_desc, {2.0, 42.8424}, false);
  }

  norm_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), blob_desc,
      {-1.419513, -23.194343, -0.002826, 12.105519, -5.655441, -34.482001,
       16.969148, 15.921443, -9.891368, 29.649382});
  if (center) {
    norm_test_case->template BackwardCheckBlob<T>(
        GenDiffBn("beta"), one_blob_desc, {14, 41.566600});
  }
  if (scale) {
    norm_test_case->template BackwardCheckBlob<T>(
        GenDiffBn("gamma"), one_blob_desc, {5.655440, -57.284965});
  }
}
TEST_CPU_AND_GPU_OPKERNEL(NormalizationTestCase_second_piece_transpose,
                          FLOATING_DATA_TYPE_SEQ, (train)(predict),
                          (forward)(backward));

}  // namespace test

}  // namespace oneflow
