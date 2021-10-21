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
#ifndef ONEFLOW_CORE_ACTOR_ACT_EVENT_LOGGER_H_
#define ONEFLOW_CORE_ACTOR_ACT_EVENT_LOGGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/actor/act_event.pb.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class ActEventLogger final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActEventLogger);
  ~ActEventLogger() = default;

  void PrintActEventToLogDir(const ActEvent&);
  static std::string experiment_act_event_bin_filename();
  static std::string act_event_bin_filename();

 private:
  static const std::string experiment_prefix_;
  static const std::string act_event_bin_filename_;
  static const std::string act_event_txt_filename_;

  friend class Global<ActEventLogger>;
  ActEventLogger(bool is_experiment_phase);

  PersistentOutStream bin_out_stream_;
  PersistentOutStream txt_out_stream_;
};
void ParseActEvents(const std::string& act_event_filepath,
                    std::list<std::unique_ptr<ActEvent>>* act_events);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACT_EVENT_LOGGER_H_
