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
#ifndef ONEFLOW_USER_SUMMARY_EVENTS_WRITER_H_
#define ONEFLOW_USER_SUMMARY_EVENTS_WRITER_H_

#include "oneflow/core/persistence/posix/posix_file_system.h"
#include "oneflow/core/common/util.h"
#include "oneflow/user/summary/crc32c.h"
#include "oneflow/core/summary/event.pb.h"

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
  void Close();

  void AppendQueue(std::unique_ptr<Event> event);
  void FileFlush();

 private:
  Maybe<void> TryToInit();
  inline static void EncodeHead(char* head, size_t size);
  inline static void EncodeTail(char* tail, const char* data, size_t size);

  bool is_inited_;
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

#endif  // ONEFLOW_USER_SUMMARY_EVENTS_WRITER_H_
