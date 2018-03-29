#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void NormalizationTestCase_single_number_train(
    OpKernelTestCase<device_type>* norm_test_case, const std::string& job_type,
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

  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("inputs", one_blob_desc, {1});

  Blob* mmean_blob =
      norm_test_case->template InitBlob<T>("moving_mean", one_blob_desc, {0});
  Blob* mvariance_blob = norm_test_case->template InitBlob<T>(
      "moving_variance", one_blob_desc, {1});

  if (center) norm_test_case->template InitBlob<T>("beta", one_blob_desc, {0});
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {1});
  }
  norm_test_case->template InitBlob<T>(GenDiffBn("outputs"), one_blob_desc,
                                       {1});

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
  norm_test_case->template ForwardCheckBlob<T>("inv_var", one_blob_desc,
                                               {31.622776});
  norm_test_case->template ForwardCheckBlob<T>("outputs", one_blob_desc, {0});
  norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                               {0.01}, false);
  norm_test_case->template ForwardCheckBlob<T>("moving_variance", one_blob_desc,
                                               {0.99}, false);

  norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("inputs"),
                                                one_blob_desc, {31.622776});
  if (center) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("beta"),
                                                  one_blob_desc, {1});
  }
  if (scale) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("gamma"),
                                                  one_blob_desc, {0});
  }
}

TEST_CPU_AND_GPU_OPKERNEL(NormalizationTestCase_single_number_train,
                          FLOATING_DATA_TYPE_SEQ, (train), (forward)(backward));

template<DeviceType device_type, typename T>
void NormalizationTestCase_single_number_predict(
    OpKernelTestCase<device_type>* norm_test_case, const std::string& job_type,
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

  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("inputs", one_blob_desc, {1});

  Blob* mmean_blob =
      norm_test_case->template InitBlob<T>("moving_mean", one_blob_desc, {0});
  Blob* mvariance_blob = norm_test_case->template InitBlob<T>(
      "moving_variance", one_blob_desc, {1});

  if (center) norm_test_case->template InitBlob<T>("beta", one_blob_desc, {0});
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {1});
  }
  norm_test_case->template InitBlob<T>(GenDiffBn("outputs"), one_blob_desc,
                                       {1});

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
  norm_test_case->template ForwardCheckBlob<T>("inv_var", one_blob_desc,
                                               {0.9995003});
  norm_test_case->template ForwardCheckBlob<T>("outputs", one_blob_desc,
                                               {0.9995003});
  norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                               {0}, false);
  norm_test_case->template ForwardCheckBlob<T>("moving_variance", one_blob_desc,
                                               {1}, false);

  norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("inputs"),
                                                one_blob_desc, {31.622776});
  if (center) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("beta"),
                                                  one_blob_desc, {1});
  }
  if (scale) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("gamma"),
                                                  one_blob_desc, {0});
  }
}

TEST_CPU_AND_GPU_OPKERNEL(NormalizationTestCase_single_number_predict,
                          FLOATING_DATA_TYPE_SEQ, (predict),
                          (forward)(backward));

template<DeviceType device_type, typename T>
void NormalizationTestCase_first_piece_train(
    OpKernelTestCase<device_type>* norm_test_case, const std::string& job_type,
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

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 5}), GetDataType<T>::value, false, false, 1);
  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("inputs", blob_desc, {1, 2, 3, 4, 5});

  Blob* mmean_blob =
      norm_test_case->template InitBlob<T>("moving_mean", one_blob_desc, {0});
  Blob* mvariance_blob = norm_test_case->template InitBlob<T>(
      "moving_variance", one_blob_desc, {1});

  if (center) norm_test_case->template InitBlob<T>("beta", one_blob_desc, {0});
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {10});
  }
  norm_test_case->template InitBlob<T>(GenDiffBn("outputs"), blob_desc,
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
      "outputs", blob_desc, {-14.13860, -7.06930, 0, 7.06930, 14.13860});
  norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                               {3.0}, false);
  norm_test_case->template ForwardCheckBlob<T>("moving_variance", one_blob_desc,
                                               {2.0}, false);

  norm_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("inputs"), blob_desc,
      {7.06930, 14.13860, 14.13860, 42.41580, 21.20790});
  if (center) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("beta"),
                                                  one_blob_desc, {14});
  }
  if (scale) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("gamma"),
                                                  one_blob_desc, {5.655440});
  }
}

TEST_CPU_AND_GPU_OPKERNEL(NormalizationTestCase_first_piece_train,
                          FLOATING_DATA_TYPE_SEQ, (train), (forward)(backward));

template<DeviceType device_type, typename T>
void NormalizationTestCase_first_piece_predict(
    OpKernelTestCase<device_type>* norm_test_case, const std::string& job_type,
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

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 5}), GetDataType<T>::value, false, false, 1);
  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("inputs", blob_desc, {1, 2, 3, 4, 5});

  Blob* mmean_blob =
      norm_test_case->template InitBlob<T>("moving_mean", one_blob_desc, {3.0});
  Blob* mvariance_blob = norm_test_case->template InitBlob<T>(
      "moving_variance", one_blob_desc, {2.0});

  if (center) norm_test_case->template InitBlob<T>("beta", one_blob_desc, {0});
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {10});
  }
  norm_test_case->template InitBlob<T>(GenDiffBn("outputs"), blob_desc,
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
      "outputs", blob_desc, {-14.13860, -7.06930, 0, 7.06930, 14.13860});
  norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                               {3.0}, false);
  norm_test_case->template ForwardCheckBlob<T>("moving_variance", one_blob_desc,
                                               {2.0}, false);

  norm_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("inputs"), blob_desc,
      {7.06930, 14.13860, 14.13860, 42.41580, 21.20790});
  if (center) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("beta"),
                                                  one_blob_desc, {14});
  }
  if (scale) {
    norm_test_case->template BackwardCheckBlob<T>(GenDiffBn("gamma"),
                                                  one_blob_desc, {5.655440});
  }
}

TEST_CPU_AND_GPU_OPKERNEL(NormalizationTestCase_first_piece_predict,
                          FLOATING_DATA_TYPE_SEQ, (predict),
                          (forward)(backward));

template<DeviceType device_type, typename T>
void NormalizationTestCase_second_piece_train_different_preblob(
    OpKernelTestCase<device_type>* norm_test_case, const std::string& job_type,
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

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 5}), GetDataType<T>::value, false, false, 1);
  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("inputs", blob_desc, {1, 2, 3, 4, 5});

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
  norm_test_case->template InitBlob<T>(GenDiffBn("outputs"), blob_desc,
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
      "outputs", blob_desc, {-14.13860, -7.06930, 0, 7.06930, 14.13860});
  norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                               {3.0}, false);
  norm_test_case->template ForwardCheckBlob<T>("moving_variance", one_blob_desc,
                                               {2.0}, false);

  norm_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("inputs"), blob_desc,
      {7.06930, 14.13860, 14.13860, 42.41580, 21.20790});
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
    OpKernelTestCase<device_type>* norm_test_case, const std::string& job_type,
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

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 5}), GetDataType<T>::value, false, false, 1);
  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("inputs", blob_desc, {1, 2, 3, 4, 5});

  Blob* mmean_blob =
      norm_test_case->template InitBlob<T>("moving_mean", one_blob_desc, {3.0});
  Blob* mvariance_blob = norm_test_case->template InitBlob<T>(
      "moving_variance", one_blob_desc, {2.0});

  if (center) norm_test_case->template InitBlob<T>("beta", one_blob_desc, {1});
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {10});
  }
  norm_test_case->template InitBlob<T>(GenDiffBn("outputs"), blob_desc,
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
      "outputs", blob_desc,
      {1 - 1.413860, 1 - 0.706930, 1, 1.706930, 2.413860});
  norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                               {3.0}, false);
  norm_test_case->template ForwardCheckBlob<T>("moving_variance", one_blob_desc,
                                               {2.0}, false);

  norm_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("inputs"), blob_desc,
      {0.706930, 1.413860, 1.413860, 4.241580, 2.120790});
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
    OpKernelTestCase<device_type>* norm_test_case, const std::string& job_type,
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

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 5}), GetDataType<T>::value, false, false, 1);
  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("inputs", blob_desc, {1, 2, 3, 4, 5});

  Blob* mmean_blob =
      norm_test_case->template InitBlob<T>("moving_mean", one_blob_desc, {3.0});
  Blob* mvariance_blob = norm_test_case->template InitBlob<T>(
      "moving_variance", one_blob_desc, {2.0});

  if (center) norm_test_case->template InitBlob<T>("beta", one_blob_desc, {1});
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {10});
  }
  norm_test_case->template InitBlob<T>(GenDiffBn("outputs"), blob_desc,
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
      "outputs", blob_desc, {-14.13860, -7.06930, 0, 7.06930, 14.13860});
  norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                               {3.0}, false);
  norm_test_case->template ForwardCheckBlob<T>("moving_variance", one_blob_desc,
                                               {2.0}, false);

  norm_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("inputs"), blob_desc,
      {7.06930, 14.13860, 14.13860, 42.41580, 21.20790});
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
    OpKernelTestCase<device_type>* norm_test_case, const std::string& job_type,
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

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 5}), GetDataType<T>::value, false, false, 1);
  BlobDesc* one_blob_desc =
      new BlobDesc(Shape({1}), GetDataType<T>::value, false, false, 1);
  norm_test_case->template InitBlob<T>("inputs", blob_desc, {1, 2, 3, 4, 5});

  Blob* mmean_blob =
      norm_test_case->template InitBlob<T>("moving_mean", one_blob_desc, {3.0});
  Blob* mvariance_blob = norm_test_case->template InitBlob<T>(
      "moving_variance", one_blob_desc, {2.0});

  if (center) norm_test_case->template InitBlob<T>("beta", one_blob_desc, {1});
  if (scale) {
    norm_test_case->template InitBlob<T>("gamma", one_blob_desc, {10});
  }
  norm_test_case->template InitBlob<T>(GenDiffBn("outputs"), blob_desc,
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
      "outputs", blob_desc, {-1.413860, -0.706930, 0, 0.706930, 1.413860});
  norm_test_case->template ForwardCheckBlob<T>("moving_mean", one_blob_desc,
                                               {3.0}, false);
  norm_test_case->template ForwardCheckBlob<T>("moving_variance", one_blob_desc,
                                               {2.0}, false);

  norm_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("inputs"), blob_desc,
      {0.706930, 1.413860, 1.413860, 4.241580, 2.120790});
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

}  // namespace test

}  // namespace oneflow
