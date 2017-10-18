#include "oneflow/core/persistence/data_set_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/str_util.h"
#include "opencv2/opencv.hpp"

namespace oneflow {

uint8_t DataSetUtil::ValidateRecordMeta(const Record& buffer) {
  uint8_t check_sum = 0;
  int meta_len = FlexibleSizeOf<Record>(0);
  for (int i = 0; i < meta_len; ++i) {
    check_sum += reinterpret_cast<const char*>(&buffer)[i];
  }
  return check_sum;
}

uint8_t DataSetUtil::ValidateRecord(const Record& buffer) {
  return ValidateRecordMeta(buffer);
}

std::unique_ptr<Record, decltype(&free)> DataSetUtil::NewRecord(
    const std::string& key, size_t value_buf_len, DataType dtype,
    DataCompressType dctype, const std::function<void(char* buff)>& Fill) {
  size_t value_offset = RoundUp(key.size(), 8);
  auto buffer = FlexibleMalloc<Record>(value_buf_len + value_offset);
  buffer->data_type_ = dtype;
  buffer->data_compress_type_ = dctype;
  buffer->key_len_ = key.size();
  buffer->value_offset_ = value_offset;
  memset(buffer->data_, 0, value_offset);
  key.copy(buffer->mut_key_buffer(), key.size());
  if (value_buf_len) { Fill(const_cast<char*>(buffer->mut_value_buffer())); }
  UpdateRecordCheckSum(buffer.get());
  return buffer;
}

void DataSetUtil::UpdateRecordMetaCheckSum(Record* buffer) {
  uint8_t meta_check_sum = 0;
  const int meta_len = FlexibleSizeOf<Record>(0);
  for (int i = 0; i < meta_len; ++i) {
    meta_check_sum += reinterpret_cast<char*>(buffer)[i];
  }
  meta_check_sum -= buffer->meta_check_sum_;
  buffer->meta_check_sum_ = -meta_check_sum;
}

void DataSetUtil::UpdateRecordCheckSum(Record* buffer) {
  uint8_t data_check_sum = 0;
  for (int i = 0; i < buffer->len_; ++i) { data_check_sum += buffer->data_[i]; }
  buffer->data_check_sum_ = -data_check_sum;
  UpdateRecordMetaCheckSum(buffer);
}

std::unique_ptr<DataSetHeader> DataSetUtil::CreateHeader(
    const std::string& type, uint32_t data_item_count,
    const std::vector<uint32_t>& dim_array) {
  std::unique_ptr<DataSetHeader> header(new DataSetHeader);
  CHECK(type.size() <= sizeof(header->type_));
  type.copy(header->type_, type.size(), 0);
  header->data_item_count_ = data_item_count;
  CHECK(dim_array.size() <= sizeof(header->dim_array_));
  header->dim_array_size_ = dim_array.size();
  memset(header->dim_array_, 0, sizeof(header->dim_array_));
  for (int i = 0; i < dim_array.size(); ++i) {
    header->dim_array_[i] = dim_array[i];
  }
  UpdateHeaderCheckSum(header.get());
  CHECK(!ValidateHeader(*header));
  return header;
}

uint32_t DataSetUtil::ValidateHeader(const DataSetHeader& header) {
  static_assert(!(sizeof(DataSetHeader) % sizeof(uint32_t)), "no alignment");
  uint32_t check_sum = 0;
  int len = sizeof(DataSetHeader) / sizeof(uint32_t);
  for (int i = 0; i < len; ++i) {
    check_sum += reinterpret_cast<const uint32_t*>(&header)[i];
  }
  return check_sum;
}

void DataSetUtil::UpdateHeaderCheckSum(DataSetHeader* header) {
  static_assert(!(sizeof(DataSetHeader) % sizeof(uint32_t)), "no alignment");
  uint32_t check_sum = 0;
  int len = sizeof(DataSetHeader) / sizeof(uint32_t);
  for (int i = 0; i < len; ++i) {
    check_sum += reinterpret_cast<uint32_t*>(header)[i];
  }
  check_sum -= header->check_sum_;
  header->check_sum_ = -check_sum;
}

std::unique_ptr<Record, decltype(&free)> DataSetUtil::CreateLabelItem(
    const std::string& key, uint32_t label_index) {
  auto buffer = NewRecord(
      key, sizeof(uint32_t), DataType::kUInt32,
      [=](char* data) { *reinterpret_cast<uint32_t*>(data) = label_index; });
  return buffer;
}

std::unique_ptr<Record, decltype(&free)> DataSetUtil::CreateImageItem(
    const std::string& img_file_path) {
  cv::Mat img = cv::imread(img_file_path);
  std::vector<unsigned char> raw_buf;
  std::vector<int> param{CV_IMWRITE_JPEG_QUALITY, 95};
  cv::imencode(".jpg", img, raw_buf, param);
  auto buffer = NewRecord(
      img_file_path, raw_buf.size(), DataType::kChar, DataCompressType::kJpeg,
      [&](char* data) { memcpy(data, raw_buf.data(), raw_buf.size()); });
  return buffer;
}

void DataSetUtil::GetFilePaths(
    const std::vector<std::string>& image_directories,
    std::vector<std::string>* img_file_paths,
    std::unordered_map<std::string, uint32_t>* file_path2label_idx) {
  for (int i = 0; i < image_directories.size(); ++i) {
    const auto& dir = image_directories[i];
    for (const auto& file_name : LocalFS()->ListDir(dir)) {
      const auto& file_path = dir + "/" + file_name;
      img_file_paths->push_back(file_path);
      (*file_path2label_idx)[file_path] = i;
    }
  }
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(img_file_paths->begin(), img_file_paths->end(),
               std::default_random_engine(seed));
}

void DataSetUtil::SaveLabels(
    const std::vector<std::string>& image_directories,
    const std::vector<std::string>& img_file_paths,
    const std::unordered_map<std::string, uint32_t>& file_path2label_idx,
    const std::string& output_dir) {
  std::vector<std::string> label_names(image_directories.size());
  for (int i = 0; i < image_directories.size(); ++i) {
    const auto& dir = image_directories[i];
    label_names[i] = basename(const_cast<char*>(dir.c_str()));
  }
  PersistentOutStream label_stream(LocalFS(), JoinPath(output_dir, "labels"));
  auto header = DataSetUtil::CreateHeader("label", img_file_paths.size(), {1});
  label_stream << *header;
  for (const auto& file_path : img_file_paths) {
    auto item = DataSetUtil::CreateLabelItem(file_path,
                                             file_path2label_idx.at(file_path));
    label_stream << *item;
  }
}

void DataSetUtil::SaveFeatures(const std::vector<std::string>& img_file_paths,
                               uint32_t width, uint32_t height,
                               const std::string& output_dir) {
  PersistentOutStream feature_stream(LocalFS(),
                                     JoinPath(output_dir, "features"));
  auto header = DataSetUtil::CreateHeader("feature", img_file_paths.size(),
                                          {3, width, height});
  feature_stream << *header;
  for (int i = 0; i < img_file_paths.size(); ++i) {
    const auto& file_path = img_file_paths.at(i);
    auto buffer = DataSetUtil::CreateImageItem(file_path);
    feature_stream << *buffer;
  }
}

void DataSetUtil::CreateDataSetFiles(
    const std::vector<std::string>& image_directories, uint32_t width,
    uint32_t height, const std::string& output_dir) {
  //  LocalFS()->CreateDirIfNotExist(output_dir);
  std::vector<std::string> img_file_paths;
  std::unordered_map<std::string, uint32_t> file_path2label_idx;
  GetFilePaths(image_directories, &img_file_paths, &file_path2label_idx);
  SaveLabels(image_directories, img_file_paths, file_path2label_idx,
             output_dir);
  SaveFeatures(img_file_paths, width, height, output_dir);
}

}  // namespace oneflow
