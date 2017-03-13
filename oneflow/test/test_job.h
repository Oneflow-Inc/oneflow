#ifndef TEST_TEST_JOB_H_
#define TEST_TEST_JOB_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "caffe.pb.h"
#include "common/filler.h"
#include "context/id_map.h"
#include "context/one.h"
#include "layers/base_layer.h"
#include "proto_io.h"

namespace caffe {
// defined types for typed test
typedef ::testing::Types<float, double> TestDtypes;
// set device id in senario "single machine single card"
const int32_t kDeviceId = 1;
// global seed for RNG
const size_t kSeed = 1000ULL;
// sovler location
#define SOLVER_LOCATION \
  "C:/Users/jiyuan/Documents/Project/jiyuan/thor/test/lenet_solver_light_with_lrn.prototxt"
// database for training and testing location
#define DATABASE_LOCATION std::string(\
  "C:/Users/v-xud/Documents/v-xud/caffe_parallel_win/caffe/data/mnist/")
// train_image filename
#define TRAIN_IMAGE_FILENAME std::string("train-images-idx3-ubyte")
// train_label filename
#define TRAIN_LABEL_FILENAME std::string("train-labels-idx1-ubyte")
// test_image filename
#define TEST_IMAGE_FILENAME std::string("t10k-images-idx3-ubyte")
// test_label filename
#define TEST_LABEL_FILENAME std::string("t10k-labels-idx1-ubyte")
}  // namespace caffe
#endif  // TEST_TEST_JOB_H_
