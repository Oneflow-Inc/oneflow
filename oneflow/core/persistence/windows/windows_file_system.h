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
#ifndef ONEFLOW_CORE_PERSISTENCE_WINDOWS_WINDOWS_FILE_SYSTEM_H_
#define ONEFLOW_CORE_PERSISTENCE_WINDOWS_WINDOWS_FILE_SYSTEM_H_

#include "oneflow/core/persistence/file_system.h"

#ifdef PLATFORM_WINDOWS

#include <Windows.h>

namespace oneflow {

namespace fs {

class WindowsFileSystem final : public FileSystem {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WindowsFileSystem);
  WindowsFileSystem() = default;
  ~WindowsFileSystem() = default;

  void NewRandomAccessFile(const std::string& fname,
                           std::unique_ptr<RandomAccessFile>* result) override;

  void NewWritableFile(const std::string& fname, std::unique_ptr<WritableFile>* result) override;

  void NewAppendableFile(const std::string& fname, std::unique_ptr<WritableFile>* result) override;

  bool FileExists(const std::string& fname) override;

  std::vector<std::string> ListDir(const std::string& dir) override;

  void DelFile(const std::string& fname) override;

  void CreateDir(const std::string& dirname) override;

  void DeleteDir(const std::string& dirname) override;

  uint64_t GetFileSize(const std::string& fname) override;

  void RenameFile(const std::string& old_name, const std::string& new_name) override;

  bool IsDirectory(const std::string& fname) override;

  static std::wstring Utf8ToWideChar(const std::string& utf8str) {
    int size_required =
        MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), (int)utf8str.size(), NULL, 0);
    std::wstring ws_translated_str(size_required, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), (int)utf8str.size(), &ws_translated_str[0],
                        size_required);
    return ws_translated_str;
  }

  static std::string WideCharToUtf8(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int size_required =
        WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string utf8_translated_str(size_required, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), &utf8_translated_str[0],
                        size_required, NULL, NULL);
    return utf8_translated_str;
  }

 private:
};

}  // namespace fs

}  // namespace oneflow

#endif  // PLATFORM_WINDOWS

#endif  // ONEFLOW_CORE_PERSISTENCE_WINDOWS_WINDOWS_FILE_SYSTEM_H_
