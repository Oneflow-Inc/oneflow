#ifndef ONEFLOW_CUSTOMIZED_SUMMARY_EVENTS_WRITER_H_
#define ONEFLOW_CUSTOMIZED_SUMMARY_EVENTS_WRITER_H_

#include "oneflow/core/persistence/posix/posix_file_system.h"
#include "oneflow/core/common/util.h"
#include "oneflow/customized/summary/crc32c.h"
#include "oneflow/customized/summary/event.pb.h"

#include <time.h>
#include <mutex>

namespace oneflow {

namespace summary {

#define MAX_QUEUE_NUM 10
#define FLUSH_TIME 3 * 60 * 1000 * 1000
#define FILE_VERSION "brain.Event:3"
const size_t kHeadSize = sizeof(uint64_t) + sizeof(uint32_t);
const size_t kTailSize = sizeof(uint32_t);

class EventsWriter {
 public:
  EventsWriter();
  ~EventsWriter();

  Maybe<void> Init(const std::string& logdir);
  void WriteEvent(const Event& event);
  void Flush();
  Maybe<void> Close();

  void AppendQueue(std::unique_ptr<Event> event);
  void FileFlush();

 private:
  Maybe<void> TryToInit();
  inline static void EncodeHead(char* head, size_t size);
  inline static void EncodeTail(char* tail, const char* data, size_t size);

  std::string log_dir_;
  std::string filename_;
  std::unique_ptr<fs::FileSystem> file_system_;
  std::unique_ptr<fs::WritableFile> writable_file_;
  uint64_t last_flush_time_;
  std::vector<std::unique_ptr<Event>> event_queue_;
  std::mutex queue_mutex;
  OF_DISALLOW_COPY(EventsWriter);
};

void EventsWriter::EncodeHead(char* head, size_t size) {
  memcpy(head, &size, sizeof(size));
  uint32_t value = MaskCrc32(GetCrc32(head, sizeof(uint64_t)));
  memcpy(head + sizeof(uint64_t), &value, sizeof(value));
}

void EventsWriter::EncodeTail(char* tail, const char* data, size_t size) {
  uint32_t value = MaskCrc32(GetCrc32(data, size));
  memcpy(tail, &value, sizeof(value));
}

}  // namespace summary

}  // namespace oneflow

#endif
