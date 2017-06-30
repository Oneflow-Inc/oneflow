#include "oneflow/core/persistence/persistent_circular_line_reader.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

PersistentCircularLineReader::PersistentCircularLineReader(
    const std::string& file_path) {
  tensorflow::Env* env = tensorflow::Env::Default();
  TF_CHECK_OK(env->NewRandomAccessFile(file_path, &file_));
  in_.reset(new tensorflow::io::InputBuffer(file_.get(), 64 * 1024 * 1024));
}

void PersistentCircularLineReader::ReadLine(std::string* line) {
  tensorflow::Status status = in_->ReadLine(line);
  if (status.code() == tensorflow::error::OUT_OF_RANGE) {
    in_->Seek(0);
    status = in_->ReadLine(line);
  }
  TF_CHECK_OK(status);
}

}  // namespace oneflow
