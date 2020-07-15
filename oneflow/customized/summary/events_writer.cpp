#include "oneflow/customized/summary/events_writer.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/customized/summary/env_time.h"

namespace oneflow {

namespace summary {

EventsWriter::EventsWriter() {}

EventsWriter::~EventsWriter() { Close(); }

Maybe<void> EventsWriter::Init(const std::string& logdir) {
  file_system_ = std::make_unique<fs::PosixFileSystem>();
  file_system_->RecursivelyCreateDirIfNotExist(logdir);
  log_dir_ = logdir;
  TryToInit();
  last_flush_time_ = CurrentMircoTime();
  return Maybe<void>::Ok();
}

Maybe<void> EventsWriter::TryToInit() {
  if (!filename_.empty()) {
    if (!file_system_->FileExists(filename_)) {
      LOG(WARNING) << "Event log file was lost, attempting create a new log file!";
    } else {
      return Maybe<void>::Ok();
    }
  }

  int32_t current_time = CurrentSecondTime();
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

void EventsWriter::AppendQueue(std::unique_ptr<Event> event) {
  queue_mutex.lock();
  event_queue_.emplace_back(std::move(event));
  queue_mutex.unlock();
  if (event_queue_.size() > MAX_QUEUE_NUM || CurrentMircoTime() - last_flush_time_ > FLUSH_TIME) {
    Flush();
  }
}

void EventsWriter::Flush() {
  queue_mutex.lock();
  for (const std::unique_ptr<Event>& e : event_queue_) { WriteEvent(*e); }
  event_queue_.clear();
  queue_mutex.unlock();
  FileFlush();
  last_flush_time_ = CurrentMircoTime();
}

void EventsWriter::WriteEvent(const Event& event) {
  std::string event_str;
  event.AppendToString(&event_str);
  if (!TryToInit().IsOk()) {
    LOG(ERROR) << "Write failed because file could not be opened.";
    return;
  }
  if (writable_file_ == nullptr) {
    LOG(WARNING) << "Log file is closed!";
    return;
  }

  char head[kHeadSize];
  char tail[kTailSize];
  EncodeHead(head, event_str.size());
  EncodeTail(tail, event_str.data(), event_str.size());
  writable_file_->Append(head, sizeof(head));
  writable_file_->Append(event_str.data(), event_str.size());
  writable_file_->Append(tail, sizeof(tail));
  FileFlush();
}

void EventsWriter::FileFlush() {
  if (writable_file_ == nullptr) { return; }
  writable_file_->Flush();
}

Maybe<void> EventsWriter::Close() {
  queue_mutex.unlock();
  Flush();
  if (writable_file_ != nullptr) {
    writable_file_->Close();
    writable_file_.reset(nullptr);
  }
  return Maybe<void>::Ok();
}

}  // namespace summary

}  // namespace oneflow
