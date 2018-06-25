#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/job/job_desc.h"
#include <cstring>

namespace oneflow {

PersistentInStream::PersistentInStream(fs::FileSystem* fs,
                                       const std::vector<std::string>& file_paths, bool cyclic,
                                       bool with_local_copy) {
  stream_buffer_filler_.reset(new StreamBufferFiller(fs, file_paths, 0, cyclic, with_local_copy));
  buffer_.resize(Global<JobDesc>::Get()->persistence_buf_byte() + 1);
  cur_buf_begin_ = buffer_.data();
  cur_buf_end_ = buffer_.data();
  *cur_buf_end_ = '\0';
}

PersistentInStream::PersistentInStream(fs::FileSystem* fs, const std::string& file_path,
                                       uint64_t offset, bool cyclic, bool with_local_copy) {
  std::vector<std::string> file_paths;
  file_paths.emplace_back(file_path);
  stream_buffer_filler_.reset(
      new StreamBufferFiller(fs, file_paths, offset, cyclic, with_local_copy));
  buffer_.resize(Global<JobDesc>::Get()->persistence_buf_byte() + 1);
  cur_buf_begin_ = buffer_.data();
  cur_buf_end_ = buffer_.data();
  *cur_buf_end_ = '\0';
}

PersistentInStream::PersistentInStream(fs::FileSystem* fs, const std::string& file_path,
                                       uint64_t offset)
    : PersistentInStream(fs, file_path, offset, false, false) {}

int32_t PersistentInStream::ReadLine(std::string* l) {
  if (IsEof()) { return -1; }
  l->clear();
  while (*cur_buf_begin_ != '\n') {
    if (cur_buf_begin_ == cur_buf_end_) {
      UpdateBuffer();
      if (cur_buf_begin_ == cur_buf_end_) {
        return 0;
      } else {
        continue;
      }
    }
    l->push_back(*cur_buf_begin_++);
  }
  ++cur_buf_begin_;
  return 0;
}

int32_t PersistentInStream::Read(char* s, size_t n) {
  if (IsEof()) { return -1; }
  while (n) {
    if (cur_buf_begin_ == cur_buf_end_) { UpdateBuffer(); }
    CHECK_LT(cur_buf_begin_, cur_buf_end_);
    int64_t copy_size = std::min(cur_buf_end_ - cur_buf_begin_, static_cast<int64_t>(n));
    std::memcpy(s, cur_buf_begin_, static_cast<size_t>(copy_size));
    s += copy_size;
    cur_buf_begin_ += copy_size;
    n -= copy_size;
  }
  return 0;
}

void PersistentInStream::UpdateBuffer() {
  CHECK_EQ(cur_buf_begin_, cur_buf_end_);
  uint64_t n = stream_buffer_filler_->UpdateBuffer(&buffer_);
  cur_buf_begin_ = buffer_.data();
  cur_buf_end_ = buffer_.data() + n;
  *cur_buf_end_ = '\0';
}

bool PersistentInStream::IsEof() const {
  return cur_buf_begin_ == cur_buf_end_ && stream_buffer_filler_->IsEof();
}
}  // namespace oneflow
