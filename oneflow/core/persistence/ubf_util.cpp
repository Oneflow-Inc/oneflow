#include "oneflow/core/persistence/ubf_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/str_util.h"
#include "opencv2/opencv.hpp"

namespace oneflow {

std::unique_ptr<UbfItem> UbfUtil::CreateLabelItem(const std::string& key,
                                                  uint32_t label_index) {
  auto ubf_item = of_make_unique<UbfItem>(
      DataType::kUInt32, DataEncodeType::kNoEncode, key, sizeof(uint32_t),
      [=](char* data) { *reinterpret_cast<uint32_t*>(data) = label_index; });
  return ubf_item;
}

std::unique_ptr<UbfItem> UbfUtil::CreateImageItem(
    const std::string& img_file_path) {
  cv::Mat img = cv::imread(img_file_path);
  std::vector<unsigned char> raw_buf;
  std::vector<int> param{CV_IMWRITE_JPEG_QUALITY, 95};
  cv::imencode(".jpg", img, raw_buf, param);
  auto ubf_item = of_make_unique<UbfItem>(
      DataType::kChar, DataEncodeType::kJpeg, img_file_path, raw_buf.size(),
      [&](char* data) { memcpy(data, raw_buf.data(), raw_buf.size()); });
  return ubf_item;
}

void UbfUtil::GetFilePaths(
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

void UbfUtil::SaveLabels(
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
  std::unique_ptr<UbfHeader> header(
      new UbfHeader("label", img_file_paths.size(), {1}));
  label_stream << *header;
  for (const std::string& file_path : img_file_paths) {
    auto item =
        UbfUtil::CreateLabelItem(file_path, file_path2label_idx.at(file_path));
    label_stream << *item;
  }
}

void UbfUtil::SaveFeatures(const std::vector<std::string>& img_file_paths,
                           uint32_t width, uint32_t height,
                           const std::string& output_dir) {
  PersistentOutStream feature_stream(LocalFS(),
                                     JoinPath(output_dir, "features"));
  std::unique_ptr<UbfHeader> header(
      new UbfHeader("feature", img_file_paths.size(), {3, width, height}));
  feature_stream << *header;
  for (int i = 0; i < img_file_paths.size(); ++i) {
    const std::string& file_path = img_file_paths.at(i);
    auto ubf_item = UbfUtil::CreateImageItem(file_path);
    feature_stream << *ubf_item;
  }
}

void UbfUtil::CreateUbfFiles(const std::vector<std::string>& image_directories,
                             uint32_t limit, uint32_t width, uint32_t height,
                             const std::string& output_dir) {
  //  LocalFS()->CreateDirIfNotExist(output_dir);
  std::vector<std::string> img_file_paths;
  std::unordered_map<std::string, uint32_t> file_path2label_idx;
  GetFilePaths(image_directories, limit, &img_file_paths, &file_path2label_idx);
  SaveLabels(image_directories, img_file_paths, file_path2label_idx,
             output_dir);
  SaveFeatures(img_file_paths, width, height, output_dir);
}

}  // namespace oneflow
