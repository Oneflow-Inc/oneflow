/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_COMMON_FILE_SYSTEM_H_
#define ONEFLOW_CORE_COMMON_FILE_SYSTEM_H_

#include <string>
#include <vector>

namespace oneflow {
namespace file_system {

class Path {
 public:
  Path() = default;
  explicit Path(const std::string& path) : path_(path) {}

  const std::string& path() const { return path_; }

  std::string basename() const;
  std::string dirname() const;

  bool exists() const;
  bool is_directory() const;
  bool is_regular_file() const;

  Path& join(const Path& rhs);

 private:
  std::string path_;
};

std::string basename(const std::string& path);
std::string dirname(const std::string& path);

bool is_directory(const std::string& path);
bool is_regular_file(const std::string& path);

std::string join_path(const std::string& lhs, const std::string& rhs);

std::vector<std::string> walk(const std::string& path);
std::vector<std::string> walk(const Path& path);

}  // namespace file_system
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_FILE_SYSTEM_H_
