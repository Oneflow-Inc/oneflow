#ifndef TEST_TEST_LENET_H
#define TEST_TEST_LENET_H
#include <fstream>
#include <string>
#include "test/test_job.h"

namespace caffe {
// Class mnist can help to read lenet data into memory. Store the necessary
// parameters in memory.
template <typename Dtype>
class Mnist {
 public:
  class MnistImage {
   public:
     explicit MnistImage(const std::string& location) {
      magic_number = image_number = row = col = 0;
      std::ifstream fin(location, std::ios::in | std::ios::binary);
      CHECK(fin);
      uint32_t *ptoval[4] = { &magic_number, &image_number, &row, &col };
      uint8_t buffer[4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          fin >> buffer[j];
          *(ptoval[i]) |= ((uint32_t)buffer[j]) << (8 * (3 - j));
        }
      }
      image_size = row * col;
      memory_count = image_size * image_number;
      memory_size = memory_count * sizeof(Dtype);

      memory_pool = reinterpret_cast<Dtype*>(malloc(memory_size));
      for (int i = 0; i < memory_count; ++i) {
        uint8_t tmp;
        fin.read(reinterpret_cast<char*>(&tmp), sizeof(uint8_t));
        memory_pool[i] = (Dtype)tmp;
      }
      fin.close();
    }
    ~MnistImage() {
      free(memory_pool);
    }

    uint32_t magic_number;
    uint32_t image_number;
    uint32_t row;
    uint32_t col;

    size_t memory_count;
    size_t memory_size;
    size_t image_size;

    Dtype* memory_pool;
  };
  class MnistLabel {
   public:
    explicit MnistLabel(const std::string& location) {
      magic_number = items = 0;
      std::ifstream fin(location, std::ios::in | std::ios::binary);
      CHECK(fin);
      uint32_t *ptoval[2] = { &magic_number, &items };
      uint8_t buffer[4];
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
          fin >> buffer[j];
          *(ptoval[i]) |= ((uint32_t)buffer[j]) << (8 * (3 - j));
        }
      }
      memory_count = items;
      memory_size = memory_count * sizeof(Dtype);
      memory_pool = reinterpret_cast<Dtype*>(malloc(memory_size));
      for (int i = 0; i < memory_count; ++i) {
        uint8_t tmp;
        fin.read(reinterpert_cast<char*(&tmp), sizeof(uint8_t));
        memory_pool[i] = (Dtype)tmp;
      }
      fin.close();
    }
    ~MnistLabel() {
      free(memory_pool);
    }

    uint32_t magic_number;
    uint32_t items;

    size_t memory_count;
    size_t memory_size;

    Dtype* memory_pool;
  };
  inline static void test() {
    get();
  }
  inline static std::shared_ptr<MnistImage>& train_image() {
    return get().train_image_;
  }
  inline static std::shared_ptr<MnistLabel>& train_label() {
    return get().train_label_;
  }
  inline static std::shared_ptr<MnistImage>& test_image() {
    return get().test_image_;
  }
  inline static std::shared_ptr<MnistLabel>& test_label() {
    return get().test_label_;
  }

 private:
  Mnist() {
    train_image_ = std::make_shared<MnistImage>(
      DATABASE_LOCATION + TRAIN_IMAGE_FILENAME);
    train_label_ = std::make_shared<MnistLabel>(
      DATABASE_LOCATION + TRAIN_LABEL_FILENAME);
    test_image_ = std::make_shared<MnistImage>(
      DATABASE_LOCATION + TEST_IMAGE_FILENAME);
    test_label_ = std::make_shared<MnistLabel>(
      DATABASE_LOCATION + TEST_LABEL_FILENAME);
  }
  inline static Mnist& get() {
    if (!sigleton_.get()) {
      sigleton_.reset(new Mnist());
    }
    return *sigleton_;
  }

  std::shared_ptr<MnistImage> train_image_;
  std::shared_ptr<MnistLabel> train_label_;

  std::shared_ptr<MnistImage> test_image_;
  std::shared_ptr<MnistLabel> test_label_;

  static std::shared_ptr<Mnist<Dtype>> sigleton_;
};

std::shared_ptr<Mnist<float>> Mnist<float>::sigleton_;
std::shared_ptr<Mnist<double>> Mnist<double>::sigleton_;

}  // namespace caffe
#endif  // TEST_TEST_LENET_H
