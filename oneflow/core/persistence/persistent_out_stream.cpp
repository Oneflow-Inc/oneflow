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
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {

PersistentOutStream::PersistentOutStream(fs::FileSystem* fs, const std::string& file_path) {
  std::string file_dir = Dirname(file_path);
  OfCallOnce(GlobalProcessCtx::LogDirEntry() + "/" + file_dir, fs,
             &fs::FileSystem::RecursivelyCreateDirIfNotExist, file_dir);
  fs->NewWritableFile(file_path, &file_);
}

PersistentOutStream::~PersistentOutStream() { file_->Close(); }

PersistentOutStream& PersistentOutStream::Write(const char* s, size_t n) {
  file_->Append(s, n);
  return *this;
}

void PersistentOutStream::Flush() { file_->Flush(); }

}  // namespace oneflow
