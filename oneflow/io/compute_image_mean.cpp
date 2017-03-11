#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <ostream>
#include "glog/logging.h"
#include "io/io.h"

int main(int argc, char* argv[]) {
  using namespace caffe;  // NOLINT(build/namespaces)
  using std::vector;

  if (argc < 2 || argc > 3) {
    LOG(INFO) << "Compute the mean_image of a set of images given by"
      " a binary file\n"
      "Usage:\n"
      "    compute_image_mean INPUT_FILE [OUTPUT_FILE]\n";
    return 1;
  }

  BinaryInputStream stream;
  if (!stream.Open(argv[1])) {
    LOG(ERROR) << "open input_file error";
    return -1;
  }

  const int images_num = stream.get_total_num();

  vector<char> buf;
  int count = 0;
  uint32_t total_num = stream.get_total_num();
  uint64_t datasize = 0;
  float *label = new float[images_num];
  float *sum_data = nullptr;
  int channel = 0;
  int height = 0;
  int width = 0;
  int mean_data_num = 0;

  LOG(INFO) << "Starting Iteration";
  while (count < total_num) {
    datasize = stream.Next(&buf, label);
    label++;
    cv::Mat res;
    cv::Mat buf(1, datasize, CV_8U, &buf[0]);
    res = cv::imdecode(buf, -1);
    if (sum_data == nullptr) {
      channel = res.channels();
      height = res.rows;
      width = res.cols;
      mean_data_num = channel*height*width;
      sum_data = new float[mean_data_num];
      memset(sum_data, 0, sizeof(float)*mean_data_num);
    }
    CHECK_EQ(channel, res.channels()) << "Incorrect channel number";
    CHECK_EQ(height, res.rows) << "Incorrect height";
    CHECK_EQ(width, res.cols) << "Incorrect width";
    // For RGB or RGBA data, swap the B and R channel:
    // OpenCV store as BGR (or BGRA) and we want RGB (or RGBA)
    std::vector<int> swap_indices;
    if (channel == 1) swap_indices = { 0 };
    if (channel == 3) swap_indices = { 2, 1, 0 };
    if (channel == 4) swap_indices = { 2, 1, 0, 3 };

    uint32_t sum_index = 0;
    for (int i = 0; i < height; ++i) {
      uchar* im_data = res.ptr<uchar>(i);
      for (int j = 0; j < width; ++j) {
        for (int k = 0; k < channel; ++k) {
          sum_index = (k*height + i)*width + j;
          sum_data[sum_index] += im_data[swap_indices[k]];
        }
        im_data += channel;
      }
    }

    res.release();
    count++;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
  CHECK_EQ(count, images_num) << "Incorrect images number";
  for (int i = 0; i < mean_data_num; ++i) {
    sum_data[i] = sum_data[i] / count;
  }
  // Write to disk
  if (argc == 3) {
    LOG(INFO) << "Write to " << argv[2];
    std::ofstream fo(argv[2], std::ios::binary);
    fo.write(reinterpret_cast<char*>(sum_data), mean_data_num*sizeof(float));
    fo.close();
  }
  const int dim = height*width;
  std::vector<float> mean_values(channel, 0.0);
  LOG(INFO) << "Number of channels: " << channel;
  for (int c = 0; c < channel; ++c) {
    for (int i = 0; i < dim; ++i) {
      mean_values[c] += sum_data[dim * c + i];
    }
    LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
  }
  stream.Close();
  return 0;
}
