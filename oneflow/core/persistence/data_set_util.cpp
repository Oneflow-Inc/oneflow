#include "oneflow/core/persistence/data_set_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/str_util.h"
#include "opencv2/opencv.hpp"

namespace oneflow {

uint8_t DataSetUtil::ValidateOfbItemMeta(const OfbItem& ofb_item) {
  uint8_t check_sum = 0;
  int meta_len = FlexibleSizeOf<OfbItem>(0);
  for (int i = 0; i < meta_len; ++i) {
    check_sum += reinterpret_cast<const char*>(&ofb_item)[i];
  }
  return check_sum;
}

uint8_t DataSetUtil::ValidateOfbItem(const OfbItem& ofb_item) {
  return ValidateOfbItemMeta(ofb_item);
}

std::unique_ptr<OfbItem, decltype(&free)> DataSetUtil::NewOfbItem(
    const std::string& key, size_t value_buf_len, DataType dtype,
    DataEncodeType detype, const std::function<void(char* buff)>& Fill) {
  size_t value_offset = RoundUpToAlignment(key.size(), 8);
  auto ofb_item = FlexibleMalloc<OfbItem>(value_buf_len + value_offset);
  ofb_item->data_type_ = dtype;
  ofb_item->data_encode_type_ = detype;
  ofb_item->key_len_ = key.size();
  ofb_item->value_offset_ = value_offset;
  memset(ofb_item->data_, 0, value_offset);
  key.copy(ofb_item->mut_key_buffer(), key.size());
  if (value_buf_len) { Fill(const_cast<char*>(ofb_item->mut_value_buffer())); }
  UpdateOfbItemCheckSum(ofb_item.get());
  return ofb_item;
}

void DataSetUtil::UpdateOfbItemMetaCheckSum(OfbItem* ofb_item) {
  uint8_t meta_check_sum = 0;
  const int meta_len = FlexibleSizeOf<OfbItem>(0);
  for (int i = 0; i < meta_len; ++i) {
    meta_check_sum += reinterpret_cast<char*>(ofb_item)[i];
  }
  meta_check_sum -= ofb_item->meta_check_sum_;
  ofb_item->meta_check_sum_ = -meta_check_sum;
}

void DataSetUtil::UpdateOfbItemCheckSum(OfbItem* ofb_item) {
  uint8_t data_check_sum = 0;
  for (int i = 0; i < ofb_item->len_; ++i) {
    data_check_sum += ofb_item->data_[i];
  }
  ofb_item->data_check_sum_ = -data_check_sum;
  UpdateOfbItemMetaCheckSum(ofb_item);
}

std::unique_ptr<OfbHeader> DataSetUtil::CreateHeader(
    const std::string& type, uint32_t data_item_count,
    const std::vector<uint32_t>& dim_array) {
  std::unique_ptr<OfbHeader> header(new OfbHeader);
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

uint32_t DataSetUtil::ValidateHeader(const OfbHeader& header) {
  static_assert(!(sizeof(OfbHeader) % sizeof(uint32_t)), "no alignment");
  uint32_t check_sum = 0;
  int len = sizeof(OfbHeader) / sizeof(uint32_t);
  for (int i = 0; i < len; ++i) {
    check_sum += reinterpret_cast<const uint32_t*>(&header)[i];
  }
  return check_sum;
}

void DataSetUtil::UpdateHeaderCheckSum(OfbHeader* header) {
  static_assert(!(sizeof(OfbHeader) % sizeof(uint32_t)), "no alignment");
  uint32_t check_sum = 0;
  int len = sizeof(OfbHeader) / sizeof(uint32_t);
  for (int i = 0; i < len; ++i) {
    check_sum += reinterpret_cast<uint32_t*>(header)[i];
  }
  check_sum -= header->check_sum_;
  header->check_sum_ = -check_sum;
}

std::unique_ptr<OfbItem, decltype(&free)> DataSetUtil::CreateLabelItem(
    const std::string& key, uint32_t label_index) {
  auto ofb_item = NewOfbItem(
      key, sizeof(uint32_t), DataType::kUInt32,
      [=](char* data) { *reinterpret_cast<uint32_t*>(data) = label_index; });
  return ofb_item;
}

std::unique_ptr<OfbItem, decltype(&free)> DataSetUtil::CreateImageItem(
    const std::string& img_file_path) {
  cv::Mat img = cv::imread(img_file_path);
  std::vector<unsigned char> raw_buf;
  std::vector<int> param{CV_IMWRITE_JPEG_QUALITY, 95};
  cv::imencode(".jpg", img, raw_buf, param);
  auto ofb_item = NewOfbItem(
      img_file_path, raw_buf.size(), DataType::kChar, DataEncodeType::kJpeg,
      [&](char* data) { memcpy(data, raw_buf.data(), raw_buf.size()); });
  return ofb_item;
}

void DataSetUtil::GetFilePaths(
    const std::vector<std::string>& image_directories, uint32_t limit,
    std::vector<std::string>* img_file_paths,
    std::unordered_map<std::string, uint32_t>* file_path2label_idx) {
  limit = std::min(limit, static_cast<uint32_t>(image_directories.size()));
  for (int i = 0; i < limit; ++i) {
    const std::string& dir = image_directories[i];
    for (const std::string& file_name : LocalFS()->ListDir(dir)) {
      const std::string& file_path = JoinPath(dir, file_name);
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
    const std::string& dir = image_directories[i];
    label_names[i] = basename(const_cast<char*>(dir.c_str()));
  }
  PersistentOutStream label_stream(LocalFS(), JoinPath(output_dir, "labels"));
  auto header = DataSetUtil::CreateHeader("label", img_file_paths.size(), {1});
  label_stream << *header;
  for (const std::string& file_path : img_file_paths) {
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
    const std::string& file_path = img_file_paths.at(i);
    auto ofb_item = DataSetUtil::CreateImageItem(file_path);
    feature_stream << *ofb_item;
  }
}

void DataSetUtil::CreateDataSetFiles(
    const std::vector<std::string>& image_directories, uint32_t limit,
    uint32_t width, uint32_t height, const std::string& output_dir) {
  //  LocalFS()->CreateDirIfNotExist(output_dir);
  std::vector<std::string> img_file_paths;
  std::unordered_map<std::string, uint32_t> file_path2label_idx;
  GetFilePaths(image_directories, limit, &img_file_paths, &file_path2label_idx);
  SaveLabels(image_directories, img_file_paths, file_path2label_idx,
             output_dir);
  SaveFeatures(img_file_paths, width, height, output_dir);
}

}  // namespace oneflow
