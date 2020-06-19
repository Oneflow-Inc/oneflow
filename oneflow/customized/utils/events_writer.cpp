#include "oneflow/customized/utils/events_writer.h"
#include <cstdio>
#include <queue>
#include "oneflow/core/common/str_util.h"
#include "oneflow/customized/utils/env_time.h"

namespace oneflow {

EventsWriter::EventsWriter(const std::string& file_prefix)
    : file_prefix_(file_prefix), max_queue_(10), flush_millis_(2 * 60 * 1000) {}

EventsWriter::~EventsWriter() {
  Close();  // Autoclose in destructor.
}

Maybe<void> EventsWriter::Init() { return InitWithSuffix(""); }

Maybe<void> EventsWriter::Initialize(const std::string& logdir,
                                     const std::string& filename_suffix) {
  file_system_ = std::make_unique<fs::PosixFileSystem>();
  file_system_->RecursivelyCreateDirIfNotExist(logdir);
  log_dir_ = logdir;
  InitWithSuffix(filename_suffix);
  last_flush_ = envtime::CurrentMircoTime();
  return Maybe<void>::Ok();
}

Maybe<void> EventsWriter::InitWithSuffix(const std::string& suffix) {
  file_suffix_ = suffix;
  return InitIfNeeded();
}

Maybe<void> EventsWriter::InitIfNeeded() {
  if (!filename_.empty()) {
    if (!file_system_->FileExists(filename_)) {
      // Todo
      return Maybe<void>::Ok();
    } else {
      return Maybe<void>::Ok();
    }
  }

  int32_t time_in_seconds = envtime::CurrentMircoTime() / 1000000;

  char hostname[255];
  CHECK_EQ(gethostname(hostname, sizeof(hostname)), 0);

  char fname[100] = {'\0'};
  snprintf(fname, 100, "%s.out.tfevents.%d.%s%s", file_prefix_.c_str(), time_in_seconds, hostname,
           file_suffix_.c_str());

  filename_ = JoinPath(log_dir_, fname);

  file_system_->NewWritableFile(filename_, &writable_file_);
  CHECK_OR_RETURN(writable_file_ != nullptr);
  {
    Event event;
    event.set_wall_time(time_in_seconds);
    event.set_file_version("brain.Event:2");
    WriteEvent(event);
    Flush();
  }
  return Maybe<void>::Ok();
}

std::string EventsWriter::FileName() {
  if (filename_.empty()) { InitIfNeeded(); }
  return filename_;
}

void EventsWriter::WriteSerializedEvent(std::string event_str) {
  if (!InitIfNeeded().IsOk()) {
    LOG(ERROR) << "Write failed because file could not be opened.";
    return;
  }
  WriteRecord(event_str);
}

Maybe<void> EventsWriter::WriteRecord(std::string data) {
  if (writable_file_ == nullptr) { return Maybe<void>::Ok(); }
  char header[kHeaderSize];
  char footer[kFooterSize];

  PopulateHeader(header, data.data(), data.size());
  PopulateFooter(footer, data.data(), data.size());
  writable_file_->Append(header, sizeof(header));
  writable_file_->Append(data.data(), data.size());
  writable_file_->Append(footer, sizeof(footer));
  Flush();
  return Maybe<void>::Ok();
}

void EventsWriter::AppendQueue(std::unique_ptr<Event> event) {
  queue_.emplace_back(std::move(event));
  if (queue_.size() > max_queue_
      || envtime::CurrentMircoTime() - last_flush_ > 1000 * flush_millis_) {
    InternalFlush();
  }
}

void EventsWriter::InternalFlush() {
  for (const std::unique_ptr<Event>& e : queue_) { WriteEvent(*e); }
  queue_.clear();
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
  if (writable_file_ != nullptr) {
    writable_file_->Close();
    writable_file_.reset(nullptr);
  }
  return Maybe<void>::Ok();
}

Maybe<void> EventsWriter::FileStillExists() { return Maybe<void>::Ok(); }

}  // namespace oneflow
