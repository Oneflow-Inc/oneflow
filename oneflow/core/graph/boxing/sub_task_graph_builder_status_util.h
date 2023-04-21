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
#ifndef ONEFLOW_CORE_GRAPH_SUB_TASK_GRAPH_BUILDER_STATUS_UTIL_H_
#define ONEFLOW_CORE_GRAPH_SUB_TASK_GRAPH_BUILDER_STATUS_UTIL_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class SubTskGphBuilderStatus;

Maybe<SubTskGphBuilderStatus> BuildSubTskGphBuilderStatus(const std::string& builder_name,
                                                          const std::string& comment);

Maybe<SubTskGphBuilderStatus> MakeComposedSubTskGphBuilderStatus(
    const std::vector<SubTskGphBuilderStatus>& status);

class SubTskGphBuilderStatus final {
 public:
  SubTskGphBuilderStatus(const std::string& builder_name, const std::string& comment)
      : builder_name_(builder_name), comment_(comment){};
  ~SubTskGphBuilderStatus() = default;

  // Getters
  const std::string& builder_name() const { return builder_name_; }
  const std::string& comment() const { return comment_; }

 private:
  std::string builder_name_;
  std::string comment_;
};

template<>
class Maybe<SubTskGphBuilderStatus> final {
 public:
  Maybe(const SubTskGphBuilderStatus& data)
      : data_or_error_(std::make_shared<SubTskGphBuilderStatus>(data)) {}
  Maybe(SubTskGphBuilderStatus&& data)
      : data_or_error_(std::make_shared<SubTskGphBuilderStatus>(std::move(data))) {}
  Maybe(const Error& error) : data_or_error_(error.stacked_error()) {}
  Maybe(const std::shared_ptr<SubTskGphBuilderStatus>& data) : data_or_error_(data) {}
  Maybe(std::shared_ptr<SubTskGphBuilderStatus>&& data) : data_or_error_(std::move(data)) {}
  Maybe(const std::shared_ptr<StackedError>& error) : data_or_error_(error) {}
  Maybe(const Maybe&) = default;
  Maybe(Maybe&& other) : data_or_error_(std::move(other.data_or_error_)) {}
  ~Maybe() = default;

  void operator=(const Maybe<SubTskGphBuilderStatus>& rhs) { data_or_error_ = rhs.data_or_error_; }
  void operator=(Maybe<SubTskGphBuilderStatus>&& rhs) {
    data_or_error_ = std::move(rhs.data_or_error_);
  }

  bool IsOk() const { return data_or_error_.template Has<SubTskGphBuilderStatus>(); }
  std::shared_ptr<SubTskGphBuilderStatus> Data_YouAreNotAllowedToCallThisFuncOutsideThisFile()
      const {
    return data_or_error_.template Get<SubTskGphBuilderStatus>();
  }
  std::shared_ptr<StackedError> stacked_error() const {
    return data_or_error_.template Get<StackedError>();
  }
  std::shared_ptr<const ErrorProto> error() const { return stacked_error()->error_proto(); }

  std::string GetSerializedError() const {
    CHECK(!IsOk());
    return GetFormatedSerializedError(this->stacked_error());
  }

  SubTskGphBuilderStatus GetDataAndSerializedStackedError(
      std::string* error_str, const SubTskGphBuilderStatus& default_for_error) const {
    if (IsOk()) {
      *error_str = StackedError().DebugString();
      return *Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
    } else {
      *error_str = this->stacked_error()->DebugString();
      return default_for_error;
    }
  }

  std::pair<SubTskGphBuilderStatus, std::shared_ptr<StackedError>> GetDataAndStackedError(
      const SubTskGphBuilderStatus& default_for_error) const {
    if (IsOk()) {
      return std::make_pair(*Data_YouAreNotAllowedToCallThisFuncOutsideThisFile(),
                            std::shared_ptr<StackedError>());
    } else {
      return std::make_pair(default_for_error, stacked_error());
    }
  }

  std::pair<std::shared_ptr<SubTskGphBuilderStatus>, std::shared_ptr<StackedError>>
  GetDataPtrAndStackedError() const {
    if (IsOk()) {
      return std::make_pair(Data_YouAreNotAllowedToCallThisFuncOutsideThisFile(),
                            std::shared_ptr<StackedError>());
    } else {
      return std::make_pair(std::shared_ptr<SubTskGphBuilderStatus>(), stacked_error());
    }
  }

  SubTskGphBuilderStatus GetOrThrow() const {
    if (!IsOk()) { ThrowError(stacked_error()); }
    return *Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }

  std::shared_ptr<SubTskGphBuilderStatus> GetPtrOrThrow() const {
    if (!IsOk()) { ThrowError(stacked_error()); }
    return Data_YouAreNotAllowedToCallThisFuncOutsideThisFile();
  }

 private:
  EitherPtr<SubTskGphBuilderStatus, StackedError> data_or_error_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_SUB_TASK_GRAPH_BUILDER_STATUS_UTIL_H_
