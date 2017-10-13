#include "oneflow/core/kernel/imagenet_util.h"
#include <libgen.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <random>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

void ImageNetUtil::GetFilePaths(
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
  unsigned seed =
  std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(img_file_paths->begin(), img_file_paths->end(),
               std::default_random_engine(seed));
}

void ImageNetUtil::SaveLabels(
    const std::vector<std::string>& image_directories,
    const std::vector<std::string>& img_file_paths,
    const std::unordered_map<std::string, uint32_t>& file_path2label_idx,
    const std::string& output_dir) {
  std::vector<std::string> label_names(image_directories.size());
  for (int i = 0; i < image_directories.size(); ++i) {
    const auto& dir = image_directories[i];
    label_names[i] = basename(const_cast<char*>(dir.c_str()));
  }
  auto label_header = DataSetUtil::CreateClassificationLabelHeader(label_names);

  auto label_header_desc =
      DataSetUtil::CreateHeaderDesc(*label_header, img_file_paths.size());

  std::vector<uint32_t> item_label_idx;
  for (const auto& file_path : img_file_paths) {
    item_label_idx.push_back(file_path2label_idx.at(file_path));
  }
  auto label_body = DataSetUtil::CreateDataSetLabel(item_label_idx);
  std::ofstream label_stream(JoinPath(output_dir, "labels"),
                             std::ofstream::out);
  label_stream << *label_header_desc << *label_header << *label_body;
}

void ImageNetUtil::SaveFeatures(const std::vector<std::string>& img_file_paths,
                                uint32_t width, uint32_t height,
                                const std::string& output_dir) {
  auto feature_header = DataSetUtil::CreateImageFeatureHeader(width, height);
  auto feature_header_desc =
      DataSetUtil::CreateHeaderDesc(*feature_header, img_file_paths.size());

  std::ofstream feature_stream(JoinPath(output_dir, "features"),
                               std::ofstream::out);
  feature_stream << *feature_header_desc << *feature_header;
  for (const auto& file_path : img_file_paths) {
    auto item = DataSetUtil::CreateImageDataItem(*feature_header, file_path);
    feature_stream << *item;
  }
}

void ImageNetUtil::CreateDataSetFiles(
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
