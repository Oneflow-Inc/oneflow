#include "oneflow/customized/utils/events_writer.h"
#include <cstdio>
#include <queue>
#include "oneflow/core/common/str_util.h"
#include "oneflow/customized/utils/env_time.h"

namespace oneflow {

EventsWriter::EventsWriter() : max_queue_(MAX_QUEUE_NUM), flush_millis_(FLUSH_MILLIS) {}

EventsWriter::~EventsWriter() { Close(); }

Maybe<void> EventsWriter::Init(const std::string& logdir) {
  file_system_ = std::make_unique<fs::PosixFileSystem>();
  file_system_->RecursivelyCreateDirIfNotExist(logdir);
  log_dir_ = logdir;
  TryToInit();
  last_flush_ = envtime::CurrentMircoTime();
  return Maybe<void>::Ok();
}

Maybe<void> EventsWriter::TryToInit() {
  if (!filename_.empty()) {
    if (!file_system_->FileExists(filename_)) {
      // Todo
      return Maybe<void>::Ok();
    } else {
      return Maybe<void>::Ok();
    }
  }

  int32_t current_time = envtime::CurrentMircoTime() / 1000000;
  char hostname[255];
  CHECK_EQ(gethostname(hostname, sizeof(hostname)), 0);

  char fname[100] = {'\0'};
  snprintf(fname, 100, "events.out.tfevents.%d.%s.v2", current_time, hostname);

  filename_ = JoinPath(log_dir_, fname);
  file_system_->NewWritableFile(filename_, &writable_file_);
  CHECK_OR_RETURN(writable_file_ != nullptr);
  {
    Event event;
    event.set_wall_time(current_time);
    event.set_file_version(FILE_VERSION);
    WriteEvent(event);
    Flush();
  }
  return Maybe<void>::Ok();
}

std::string EventsWriter::FileName() {
  if (filename_.empty()) { TryToInit(); }
  return filename_;
}

void EventsWriter::WriteSerializedEvent(const std::string& event_str) {
  if (!TryToInit().IsOk()) {
    LOG(ERROR) << "Write failed because file could not be opened.";
    return;
  }
  WriteRecord(event_str);
}

Maybe<void> EventsWriter::WriteRecord(const std::string& data) {
  if (writable_file_ == nullptr) { return Maybe<void>::Ok(); }
  char head[kHeadSize];
  char tail[kTailSize];

  EncodeHead(head, data.data(), data.size());
  EncodeTail(tail, data.data(), data.size());
  writable_file_->Append(head, sizeof(head));
  writable_file_->Append(data.data(), data.size());
  writable_file_->Append(tail, sizeof(tail));
  Flush();
  return Maybe<void>::Ok();
}

void EventsWriter::AppendQueue(std::unique_ptr<Event> event) {
  queue_mutex.lock();
  queue_.emplace_back(std::move(event));
  queue_mutex.unlock();
  if (queue_.size() > max_queue_
      || envtime::CurrentMircoTime() - last_flush_ > 1000 * flush_millis_) {
    InternalFlush();
  }
}

void EventsWriter::InternalFlush() {
  queue_mutex.lock();
  for (const std::unique_ptr<Event>& e : queue_) { WriteEvent(*e); }
  queue_.clear();
  queue_mutex.unlock();
  Flush();
  last_flush_ = envtime::CurrentMircoTime();
}

void EventsWriter::WriteEvent(const Event& event) {
  std::string event_str;
  event.AppendToString(&event_str);
  WriteSerializedEvent(event_str);
}

Maybe<void> EventsWriter::Flush() {
  CHECK_OR_RETURN(writable_file_ != nullptr);
  writable_file_->Flush();
  return Maybe<void>::Ok();
}

Maybe<void> EventsWriter::Close() {
  Maybe<void> status = Flush();
  queue_mutex.unlock();
  if (writable_file_ != nullptr) {
    writable_file_->Close();
    writable_file_.reset(nullptr);
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
