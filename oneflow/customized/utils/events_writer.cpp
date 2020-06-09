#include "oneflow/customized/utils/events_writer.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/customized/utils/env_time.h"

namespace oneflow {

EventsWriter::EventsWriter(const std::string& file_prefix)
    // TODO(jeff,sanjay): Pass in env and use that here instead of Env::Default
    : file_prefix_(file_prefix) {}

EventsWriter::~EventsWriter() {
  Close();  // Autoclose in destructor.
}

Maybe<void> EventsWriter::Init() { return InitWithSuffix(""); }

Maybe<void> EventsWriter::Initialize(const std::string& logdir,
                                     const std::string& filename_suffix) {
  file_system_ = std::make_unique<fs::PosixFileSystem>();
  if (!file_system_->IsDirectory(logdir)) { file_system_->CreateDir(logdir); }
  InitWithSuffix(filename_suffix);
  return Maybe<void>::Ok();
}

Maybe<void> EventsWriter::InitWithSuffix(const std::string& suffix) {
  file_suffix_ = suffix;
  return InitIfNeeded();
}

Maybe<void> EventsWriter::InitIfNeeded() {
  // file_system_.reset(LocalFS());
  if (!filename_.empty()) {
    if (!file_system_->FileExists(filename_)) {
      // Todo
      return Maybe<void>::Ok();
    } else {
      return Maybe<void>::Ok();
    }
  }

  int64_t time_in_seconds = envtime::NowNanos() / envtime::kMicrosToNanos / 1000000;

  char hostname[255];
  CHECK_EQ(gethostname(hostname, sizeof(hostname)), 0);

  filename_ = "dubihe.out.ofevents.1231314141.zjlab.laohu";
  /*strings::Printf("%s.out.ofevnets.%010lld.%s%s", file_prefix_.c_str(),
                  static_cast<int64_t>(time_in_seconds), hostname, file_suffix_.c_str());*/

  file_system_->NewWritableFile(filename_, &writable_file_);
  CHECK_OR_RETURN(writable_file_ != nullptr);

  {
    Event event;
    event.set_step(2);
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

void EventsWriter::WriteEvent(const Event& event) {
  std::string record;
  event.AppendToString(&record);
  WriteSerializedEvent(record);
}

Maybe<void> EventsWriter::Flush() {
  // CHECK_OR_RETURN(writable_file_ != nullptr);
  // writable_file_->Flush();
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
