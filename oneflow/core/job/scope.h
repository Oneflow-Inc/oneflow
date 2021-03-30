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
#ifndef ONEFLOW_CORE_JOB_SCOPE_H_
#define ONEFLOW_CORE_JOB_SCOPE_H_

#include "oneflow/core/job/scope.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

class OperatorConf;

namespace cfg {
class ScopeProto;
}

class Scope final {
 public:
  Scope(const Scope&) = delete;
  Scope(Scope&&) = delete;
  explicit Scope(const ScopeProto& scope_proto);
  ~Scope() = default;

  static Maybe<Scope> New(int64_t symbol_id, const ScopeProto& scope_proto);
  const Maybe<int64_t>& symbol_id() const { return symbol_id_; }
  int64_t auto_increment_id() { return ++auto_increment_id_; }
  int64_t session_id() const { return scope_proto().session_id(); }
  const std::shared_ptr<JobDesc>& job_desc_symbol() const { return job_desc_; }
  const std::shared_ptr<ParallelDesc>& device_parallel_desc_symbol() const {
    return device_parallel_desc_;
  }
  const std::shared_ptr<Scope>& parent_scope_symbol() const { return parent_scope_symbol_; }
  Maybe<cfg::ScopeProto> MakeChildScopeProto() const;

  Maybe<const JobDesc*> job_desc() const;
  Maybe<int64_t> GetParallelDescSymbolId(const OperatorConf& op_conf) const;
  Maybe<const ParallelDesc&> GetParallelDesc(const OperatorConf& op_conf) const;

  const OptMirroredParallel& opt_mirrored_parallel_conf() const {
    return scope_proto_.opt_mirrored_parallel_conf();
  }
  const ScopeProto& scope_proto() const { return scope_proto_; }

#define DEFINE_SCOPE_CONFIG_GETTER(T, func_name, field_name) \
  T func_name(const std::string& field_name) const {         \
    const AttrValue& attr_val = GetAttrValue(field_name);    \
    CHECK(attr_val.has_##field_name());                      \
    return attr_val.field_name();                            \
  }
  DEFINE_SCOPE_CONFIG_GETTER(bool, Bool, at_bool);
  DEFINE_SCOPE_CONFIG_GETTER(int64_t, Int64, at_int64);
  DEFINE_SCOPE_CONFIG_GETTER(double, Double, at_double);
  DEFINE_SCOPE_CONFIG_GETTER(const std::string&, String, at_string);

 private:
  Scope(int64_t symbol_id, const ScopeProto& scope_proto);
  Maybe<void> Init();

  const AttrValue& GetAttrValue(const std::string& attr_name) const;

  int64_t auto_increment_id_;
  Maybe<int64_t> symbol_id_;
  const ScopeProto scope_proto_;
  std::shared_ptr<JobDesc> job_desc_;
  std::shared_ptr<ParallelDesc> device_parallel_desc_;
  std::shared_ptr<ParallelDesc> host_parallel_desc_;
  std::shared_ptr<Scope> parent_scope_symbol_;
};

Maybe<int64_t> GenNewScopeSymbolId(
    int64_t old_scope_symbol_id,
    const std::function<void(std::shared_ptr<cfg::ScopeProto> new_scope)>& InitNewScopeProto);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SCOPE_H_
