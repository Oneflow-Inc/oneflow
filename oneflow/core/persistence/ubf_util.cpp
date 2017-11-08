#include "oneflow/core/persistence/ubf_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
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
  for (int i = 0; i < img_file_paths.size(); ++i) {
    const std::string& file_path = img_file_paths.at(i);
    auto ubf_item = UbfUtil::CreateImageItem(file_path);
    feature_stream << *ubf_item;
  }
}

// template<typename Out>
// void split(const std::string& s, char delim, Out result) {
//   std::stringstream ss(s);
//   std::string item;
//   while (std::getline(ss, item, delim)) { *(result++) = item; }
// }

// std::vector<std::string> split(const std::string& s, char delim) {
//   std::vector<std::string> elems;
//   split(s, delim, std::back_inserter(elems));
//   return elems;
// }

void UbfUtil::SaveFeaturesAndLabels(
    const std::vector<std::string>& img_file_paths, uint32_t width,
    uint32_t height, const std::string& output_dir) {
  PersistentOutStream feature_stream(LocalFS(),
                                     JoinPath(output_dir, "features"));
  PersistentOutStream label_stream(LocalFS(), JoinPath(output_dir, "labels"));
  std::string line;
  int q = 0;
  JobDescProto jb_desc_proto;
  auto job_conf = jb_desc_proto.mutable_job_conf();
  auto gfs_conf = job_conf->mutable_global_fs_conf();
  gfs_conf->mutable_hdfs_conf()->set_namenode("hdfs://192.168.1.11:9000");
  auto resource = jb_desc_proto.mutable_resource();
  resource->add_machine();
  JobDesc::NewSingleton(jb_desc_proto);

  for (int i = 0; i < img_file_paths.size(); ++i) {
    const std::string& file_path = img_file_paths.at(i);
    std::cout << file_path << std::endl;
    NormalPersistentInStream in_stream(GlobalFS(), file_path);
    while (in_stream.ReadLine(&line) == 0 && q < 10) {
      // std::vector<std::string> subs;
      // split(line, "\t", std::back_inserter(subs));
      // std::cout << subs[0] << ": " << subs[1] << std::endl;
      std::vector<std::string> tokens;
      std::istringstream iss(line);
      std::string token;
      while (
          std::getline(iss, token, '\t'))  // but we can specify a different one
        tokens.push_back(token);
      std::cout << tokens[0] << ": " << tokens[1] << std::endl;
      ++q;
    }
  }
  JobDesc::DeleteSingleton();
}

void UbfUtil::CreateUbfFiles(const std::vector<std::string>& image_directories,
                             uint32_t limit, uint32_t width, uint32_t height,
                             const std::string& output_dir,
                             const bool use_hadoop_stream) {
  //  LocalFS()->CreateDirIfNotExist(output_dir);
  std::vector<std::string> img_file_paths;
  if (use_hadoop_stream) {
    SaveFeaturesAndLabels(image_directories, width, height, output_dir);
  } else {
    std::unordered_map<std::string, uint32_t> file_path2label_idx;
    GetFilePaths(image_directories, limit, &img_file_paths,
                 &file_path2label_idx);
    SaveLabels(image_directories, img_file_paths, file_path2label_idx,
               output_dir);
    SaveFeatures(img_file_paths, width, height, output_dir);
  }
}

}  // namespace oneflow
