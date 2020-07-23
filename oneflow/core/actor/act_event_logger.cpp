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
#include "oneflow/core/actor/act_event_logger.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/common/protobuf.h"
#include <google/protobuf/text_format.h>

namespace oneflow {

const std::string ActEventLogger::experiment_prefix_("experiment_");
const std::string ActEventLogger::act_event_bin_filename_("act_event.bin");
const std::string ActEventLogger::act_event_txt_filename_("act_event.txt");

void ActEventLogger::PrintActEventToLogDir(const ActEvent& act_event) {
  bin_out_stream_ << act_event;
  std::string act_event_txt;
  google::protobuf::TextFormat::PrintToString(act_event, &act_event_txt);
  txt_out_stream_ << act_event_txt;
}

std::string ActEventLogger::experiment_act_event_bin_filename() {
  return experiment_prefix_ + act_event_bin_filename_;
}

std::string ActEventLogger::act_event_bin_filename() { return act_event_bin_filename_; }

ActEventLogger::ActEventLogger(bool is_experiment)
    : bin_out_stream_(LocalFS(), JoinPath(FLAGS_log_dir, (is_experiment ? experiment_prefix_ : "")
                                                             + act_event_bin_filename_)),
      txt_out_stream_(LocalFS(), JoinPath(FLAGS_log_dir, (is_experiment ? experiment_prefix_ : "")
                                                             + act_event_txt_filename_)) {}

void ParseActEvents(const std::string& act_event_filepath,
                    std::list<std::unique_ptr<ActEvent>>* act_events) {
  PersistentInStream in_stream(LocalFS(), act_event_filepath);
  int64_t act_event_size;
  while (!in_stream.ReadFully(reinterpret_cast<char*>(&act_event_size), sizeof(act_event_size))) {
    std::vector<char> buffer(act_event_size);
    CHECK(!in_stream.ReadFully(buffer.data(), act_event_size));
    auto act_event = std::make_unique<ActEvent>();
    act_event->ParseFromArray(buffer.data(), act_event_size);
    act_events->emplace_back(std::move(act_event));
  }
}
}  // namespace oneflow
