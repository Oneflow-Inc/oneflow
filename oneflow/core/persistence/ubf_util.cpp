#include "oneflow/core/persistence/ubf_util.h"
#include "base64.h"
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

void UbfUtil::SaveFeaturesAndLabels(
    const std::vector<std::string>& img_file_paths, uint32_t width,
    uint32_t height, const std::string& output_dir, uint32_t limit,
    const std::string& hadoop_namenode, const bool output_2_hadoop) {
  JobDescProto jb_desc_proto;
  auto job_conf = jb_desc_proto.mutable_job_conf();
  auto gfs_conf = job_conf->mutable_global_fs_conf();
  gfs_conf->mutable_hdfs_conf()->set_namenode(hadoop_namenode);
  auto resource = jb_desc_proto.mutable_resource();
  resource->add_machine();
  JobDesc::NewSingleton(jb_desc_proto);

  fs::FileSystem* fs;
  if (output_2_hadoop) {
    fs = GlobalFS();
  } else {
    fs = LocalFS();
  }

  PersistentOutStream feature_stream(fs,
                                     JoinPath(output_dir, "feature/part-0"));
  PersistentOutStream label_stream(fs, JoinPath(output_dir, "label/part-0"));
  std::string line;
  uint32_t q = 0;
  for (int i = 0; i < img_file_paths.size(); ++i) {
    const std::string& file_path = img_file_paths.at(i);
    std::cout << file_path << std::endl;
    NormalPersistentInStream in_stream(GlobalFS(), file_path);
    while (in_stream.ReadLine(&line) == 0 && q < limit) {
      ++q;
      if ((q % 1000) == 0) {
        if (limit == INT_MAX)
          LOG(INFO)
              << q
              << " of unknown number of images";  // << limit;  // << std::endl;
        else
          LOG(INFO) << q << " of " << limit;  // << std::endl;
      }

      std::vector<std::string> tokens;
      std::istringstream iss(line);
      std::string token;
      while (
          std::getline(iss, token, '\t'))  // but we can specify a different one
        tokens.push_back(token);
      token = base64_decode(tokens[2]);

      // save decode data for check
      // const char* bin_ptr = token.c_str();
      // std::ofstream output( "example.jpg", std::ios::out | std::ios::binary
      // );  output.write( bin_ptr, token.length() );

      try {
        // construct feature file
        const char* buffer = token.c_str();
        std::vector<char> src_vec(buffer, buffer + token.length());
        cv::Mat m_img = cv::Mat(src_vec);
        cv::Mat img = cv::imdecode(m_img, 1);
        std::vector<unsigned char> raw_buf;
        std::vector<int> param{CV_IMWRITE_JPEG_QUALITY, 95};
        cv::imencode(".jpg", img, raw_buf, param);
        auto ubf_item = of_make_unique<UbfItem>(
            DataType::kChar, DataEncodeType::kJpeg, tokens[0], raw_buf.size(),
            [&](char* data) { memcpy(data, raw_buf.data(), raw_buf.size()); });
        feature_stream << *ubf_item;

        // construct label file
        int32_t label_index = std::stoi(tokens[1]);
        auto label_ubf_item = of_make_unique<UbfItem>(
            DataType::kUInt32, DataEncodeType::kNoEncode, tokens[0],
            sizeof(uint32_t), [=](char* data) {
              *reinterpret_cast<uint32_t*>(data) = label_index;
            });
        label_stream << *label_ubf_item;
      } catch (cv::Exception& e) {
        const char* err_msg = e.what();
        LOG(INFO) << "Exception on image:" << tokens[0] << " " << tokens[1]
                  << " " << err_msg;  // << std::endl;
      }
      //++q;
    }
  }
  JobDesc::DeleteSingleton();
}

void UbfUtil::CreateUbfFiles(const std::vector<std::string>& image_directories,
                             uint32_t limit, uint32_t width, uint32_t height,
                             const std::string& output_dir,
                             const std::string& hadoop_namenode,
                             const bool output_2_hadoop) {
  //  LocalFS()->CreateDirIfNotExist(output_dir);
  std::vector<std::string> img_file_paths;
  if (hadoop_namenode.length() > 0) {
    SaveFeaturesAndLabels(image_directories, width, height, output_dir, limit,
                          hadoop_namenode, output_2_hadoop);
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
