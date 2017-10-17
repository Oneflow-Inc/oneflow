#include "oneflow/core/persistence/imagenet_util.h"
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
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
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
  std::ofstream label_stream(JoinPath(output_dir, "labels"),
                             std::ofstream::out);
  auto header = DataSetUtil::CreateHeader("label", DataType::kUInt32,
                                          img_file_paths.size(), {1});
  label_stream << *header;
  for (const auto& file_path : img_file_paths) {
    auto item = DataSetUtil::CreateLabelItem(*header,
                                             file_path2label_idx.at(file_path));
    label_stream << *item;
  }
}

void ImageNetUtil::SaveFeatures(const std::vector<std::string>& img_file_paths,
                                uint32_t width, uint32_t height,
                                const std::string& output_dir) {
  std::ofstream feature_stream(JoinPath(output_dir, "features"),
                               std::ofstream::out);
  auto header = DataSetUtil::CreateHeader(
      "feature", DataType::kChar, img_file_paths.size(), {3, width, height});
  feature_stream << *header;
  for (int i = 0; i < img_file_paths.size(); ++i) {
    const auto& file_path = img_file_paths.at(i);
    auto item = DataSetUtil::CreateImageItem(*header, file_path);
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
