#ifndef _IO_IMAGE_LOADER_H_
#define _IO_IMAGE_LOADER_H_

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <string>
#include <cstdint>
#include <map>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include "memory/blob.h"
#include "io/io.h"
#include "common/shape.h"

namespace caffe {
// helper class to do image augmentation
template <typename Dtype>
class ImageAugmenter {
public:
  ImageAugmenter() {}
  ~ImageAugmenter() {}
  void Init(Shape image_shape,
    const std::map<std::string, std::string> &kwargs);
  void Process(const cv::Mat &cv_img, Dtype* transformed_data);

private:
  // For data pre-processing, we can do simple scaling and subtracting the
  // data mean, if provided. Note that the mean subtraction is always carried
  // out before scaling.
  Dtype scale_{ 1 };
  // Specify if we want to randomly mirror data.
  bool mirror_{ false };
  // Specify if we would like to randomly crop an image.
  uint32_t crop_size_{ 0 };
  // mean_file and mean_value cannot be specified at the same time
  // std::string mean_file_;
  bool has_mean_file_{ false };
  std::vector<Dtype> data_mean_;
  // mean values
  std::vector<Dtype> mean_values_;

  int phase_{ Phase::TRAIN };

  // random engine
  std::mt19937 rnd_;
  // Generates a random integer from Uniform({0, 1, ..., n-1}).
  int Rand(int n) {
    std::uniform_int_distribution<int> uniform(0, n-1);
    return uniform(rnd_);
  }

  // shape
  int64_t channel_;
  int64_t height_;
  int64_t width_;
};

// image loader
template <typename Dtype>
class ImageLoader {
public:
  ImageLoader() {}
  ~ImageLoader() {}
  // Initialize ImageLoader with parameters of:
  // file_name, source binary file path
  // image_shape, image shape (batch_size, channel, height, width)
  // kwargs, the parameters to init ImageAugmenter
  // Returns false if file open failed.
  bool Init(const std::string file_name, Shape image_shape,
    const std::map<std::string, std::string> &kwargs);
  // Get next batch
  // Returns the num of images in this batch.
  // If the returned num is less than batch size,
  // means the ImageLoader has reached the end of the source file
  int32_t NextBatch(Blob<Dtype>* data_batch, Blob<Dtype>* label_batch);

private:
  BinaryInputStream stream_;
  // shape
  int64_t batch_size_;
  int64_t channel_;
  int64_t height_;
  int64_t width_;

  // data buf
  std::vector<char> data_buf_;

  ImageAugmenter<Dtype> img_aug_;
};

template <typename Dtype>
void ImageAugmenter<Dtype>::Init(Shape image_shape,
  const std::map<std::string, std::string> &kwargs) {
  channel_ = image_shape.channels();
  height_ = image_shape.height();
  width_ = image_shape.width();
  // parse parameters
  if (kwargs.find("scale") != kwargs.end()) {
    scale_ = std::stof(kwargs.at("scale"));
  }
  if (kwargs.find("mirror") != kwargs.end()) {
    if ((kwargs.at("mirror") == "True") || (kwargs.at("mirror") == "true")
      || (kwargs.at("mirror") == "1")) {
      mirror_ = true;
    }
  }
  if (kwargs.find("crop_size") != kwargs.end()) {
    crop_size_ = std::stoi(kwargs.at("crop_size"));
  }
  if (kwargs.find("mean_file") != kwargs.end()) {
    // get mean_file_ into memory
    // TODO(v-kayin): not tested, need a tool to compute images mean
    std::ifstream fi(kwargs.at("mean_file"), std::ios::binary);
    CHECK(fi.good()) << "open mean file failed!";
    uint32_t length = channel_*height_*width_;
    data_mean_.resize(length);
    fi.read(reinterpret_cast<char*>(&data_mean_[0]), length*sizeof(Dtype));
    CHECK(fi.gcount() == length*sizeof(Dtype)) << "mean file error.";
    fi.close();
    has_mean_file_ = true;
  }
  if (kwargs.find("mean_values") != kwargs.end()) {
    CHECK(!has_mean_file_) <<
      "Cannot specify mean_file and mean_value at the same time";
    std::stringstream ss(kwargs.at("mean_values"));
    Dtype tmp;
    while (ss >> tmp) {
      mean_values_.push_back(tmp);
    }
  }
  if (kwargs.find("phase") != kwargs.end()) {
    if (kwargs.at("phase") == "TEST" || kwargs.at("phase") == "test"
      || kwargs.at("phase") == "1") {
      phase_ = Phase::TEST;
    }
  }

  rnd_.seed(time(0));
}
template <typename Dtype>
void ImageAugmenter<Dtype>::Process(const cv::Mat &cv_img,
  Dtype* transformed_data) {
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  CHECK_EQ(channel_, img_channels);
  CHECK_LE(height_, img_height);
  CHECK_LE(width_, img_width);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const bool do_mirror = mirror_ && Rand(2);
  const bool has_mean_file = has_mean_file_;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size_);
  CHECK_GE(img_width, crop_size_);

  if (has_mean_file) {
    CHECK_EQ(img_channels, channel_);
    CHECK_EQ(img_height, height_);
    CHECK_EQ(img_width, width_);
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
      "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size_) {
    CHECK_EQ(crop_size_, height_);
    CHECK_EQ(crop_size_, width_);
    // We only do random crop when we do training.
    if (phase_ == Phase::TRAIN) {
      h_off = Rand(img_height - crop_size_ + 1);
      w_off = Rand(img_width - crop_size_ + 1);
    } else {
      h_off = (img_height - crop_size_) / 2;
      w_off = (img_width - crop_size_) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size_, crop_size_);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height_);
    CHECK_EQ(img_width, width_);
  }

  CHECK(cv_cropped_img.data);

  // For RGB or RGBA data, swap the B and R channel:
  // OpenCV store as BGR (or BGRA) and we want RGB (or RGBA)
  std::vector<int> swap_indices;
  if (img_channels == 1) swap_indices = { 0 };
  if (img_channels == 3) swap_indices = { 2, 1, 0 };
  if (img_channels == 4) swap_indices = { 2, 1, 0, 3 };

  int top_index;
  for (int h = 0; h < height_; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    // int img_index = 0;
    for (int w = 0; w < width_; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height_ + h) * width_ + (width_ - 1 - w);
        } else {
          top_index = (c * height_ + h) * width_ + w;
        }
        Dtype pixel = static_cast<Dtype>(ptr[swap_indices[c]]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - data_mean_[mean_index]) * scale_;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale_;
          } else {
            transformed_data[top_index] = pixel * scale_;
          }
        }
      }
      ptr += img_channels;
    }
  }
}

template <typename Dtype>
bool ImageLoader<Dtype>::Init(const std::string file_name, Shape image_shape,
  const std::map<std::string, std::string> &kwargs) {
  // randomly shuffle the data
  bool do_shuffle = false;
  if (kwargs.find("shuffle") != kwargs.end()) {
    std::string shuffle_para = kwargs.at("shuffle");
    std::transform(shuffle_para.begin(), shuffle_para.end(),
      shuffle_para.begin(), ::tolower);
    if (shuffle_para == "true" || shuffle_para == "1") {
      do_shuffle = true;
    }
  }
  if (!stream_.Open(file_name, do_shuffle)) {
    return false;
  }
  CHECK(image_shape.num_axes() == 4) << "shape init error.";
  batch_size_ = image_shape.num();
  channel_ = image_shape.channels();
  height_ = image_shape.height();
  width_ = image_shape.width();
  img_aug_.Init(image_shape, kwargs);
  return true;
}
template <typename Dtype>
int32_t ImageLoader<Dtype>::NextBatch(Blob<Dtype>* data_batch,
  Blob<Dtype>* label_batch) {
  data_buf_.clear();
  uint64_t datasize = 0;
  uint32_t count = 0;
  Dtype* data = data_batch->mutable_data();
  Dtype* label = label_batch->mutable_data();
  const uint32_t img_size = channel_*height_*width_;
  while (true) {
    datasize = stream_.Next(&data_buf_, label);
    if (datasize == 0) {
      break;
    }
    label++;
    // Opencv decode and augments
    cv::Mat res;
    cv::Mat buf(1, datasize, CV_8U, &data_buf_[0]);
    // -1 to keep the number of channel of the encoded image,
    // and not force gray or color.
    res = cv::imdecode(buf, -1);

    img_aug_.Process(res, data);
    data += img_size;

    res.release();

    count++;
    if (count >= batch_size_) {
      break;
    }
  }
  return count;
}
}  // namespace caffe
#endif  // _IO_IMAGE_LOADER_H_
