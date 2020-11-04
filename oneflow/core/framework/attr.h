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
#ifndef ONEFLOW_CORE_FRAMEWORK_ATTR_H_
#define ONEFLOW_CORE_FRAMEWORK_ATTR_H_

#include <glog/logging.h>
#include "oneflow/core/framework/attr.pb.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

class AttrsAccessor final {
 public:
  AttrsAccessor(const AttrsAccessor&) = default;
  AttrsAccessor(AttrsAccessor&&) = default;
  explicit AttrsAccessor(const AttrCollection& attrs) : attrs_(&attrs) {}
  ~AttrsAccessor() = default;

  bool HasAttrValue(const std::string& attr_name) const;
  Maybe<const AttrValue&> GetAttrValue(const std::string& attr_name) const;

  // Get values
  Maybe<bool> Bool(const std::string& name) const;
  Maybe<int64_t> Int64(const std::string& name) const;
  Maybe<double> Double(const std::string& name) const;
  Maybe<const std::string&> String(const std::string& name) const;
  Maybe<const PbRf<int64_t>&> ListInt64(const std::string& name) const;

 private:
  const AttrCollection& attrs() const { return *attrs_; }

  const AttrCollection* const attrs_;
};

class AttrsMutAccessor final {
 public:
  AttrsMutAccessor(const AttrsMutAccessor&) = default;
  AttrsMutAccessor(AttrsMutAccessor&&) = default;
  explicit AttrsMutAccessor(AttrCollection* attrs) : attrs_(attrs) {}
  ~AttrsMutAccessor() = default;

  AttrValue* MutAttrValue(const std::string& attr_name) const;

  // add attrs
  const AttrsMutAccessor& Bool(const std::string& name, bool val) const;
  const AttrsMutAccessor& Int64(const std::string& name, int64_t val) const;
  const AttrsMutAccessor& Double(const std::string& name, double val) const;
  const AttrsMutAccessor& String(const std::string& name, const std::string& val) const;
  const AttrsMutAccessor& ListInt64(const std::string& name, const std::vector<int64_t>& val) const;

 private:
  const AttrCollection& attrs() const { return *attrs_; }
  AttrCollection* mut_attrs() const { return attrs_; }

  AttrCollection* const attrs_;
};

class AttrDefsAccessor final {
 public:
  AttrDefsAccessor(const AttrDefsAccessor&) = default;
  AttrDefsAccessor(AttrDefsAccessor&&) = default;
  explicit AttrDefsAccessor(const AttrDefCollection& def) : def_(&def) {}
  ~AttrDefsAccessor() = default;

  const AttrDefCollection& attr_defs() const { return *def_; }

  bool HasAttrDef(const std::string& attr_name) const;
  Maybe<const AttrDef&> GetAttrDef(const std::string& attr_name) const;

  // Get Default values
  Maybe<bool> Bool(const std::string& name) const;
  Maybe<int64_t> Int64(const std::string& name) const;
  Maybe<double> Double(const std::string& name) const;
  Maybe<const std::string&> String(const std::string& name) const;
  Maybe<const PbRf<int64_t>&> ListInt64(const std::string& name) const;

 private:
  const AttrDefCollection& def() const { return *def_; }

  const AttrDefCollection* const def_;
};

class AttrDefsMutAccessor {
 public:
  AttrDefsMutAccessor(const AttrDefsMutAccessor&) = default;
  AttrDefsMutAccessor(AttrDefsMutAccessor&&) = default;
  explicit AttrDefsMutAccessor(AttrDefCollection* def) : def_(def) {}
  ~AttrDefsMutAccessor() = default;

  const AttrDefCollection& attr_defs() const { return *def_; }

  AttrDef* AddAttrDef(const std::string& attr_name) const;

  // Add attr defs
  const AttrDefsMutAccessor& Bool(const std::string& name, bool default_val) const {
    return Bool(name, default_val, "");
  }
  const AttrDefsMutAccessor& Int64(const std::string& name, int64_t default_val) const {
    return Int64(name, default_val, "");
  }
  const AttrDefsMutAccessor& Double(const std::string& name, double default_val) const {
    return Double(name, default_val, "");
  }
  const AttrDefsMutAccessor& String(const std::string& name, const std::string& default_val) const {
    return String(name, default_val, "");
  }
  const AttrDefsMutAccessor& ListInt64(const std::string& name,
                                       const std::vector<int64_t>& default_val) const {
    return ListInt64(name, default_val, "");
  }

  const AttrDefsMutAccessor& Bool(const std::string& name, bool default_val,
                                  const std::string& description) const;
  const AttrDefsMutAccessor& Int64(const std::string& name, int64_t default_val,
                                   const std::string& description) const;
  const AttrDefsMutAccessor& Double(const std::string& name, double default_val,
                                    const std::string& description) const;
  const AttrDefsMutAccessor& String(const std::string& name, const std::string& default_val,
                                    const std::string& description) const;

  const AttrDefsMutAccessor& ListInt64(const std::string& name,
                                       const std::vector<int64_t>& default_val,
                                       const std::string& description) const;

 private:
  const AttrDefCollection& def() const { return *def_; }
  AttrDefCollection* mut_def() const { return def_; }

  AttrDefCollection* const def_;
};

class DefaultedAttrsAccessor final {
 public:
  DefaultedAttrsAccessor(const DefaultedAttrsAccessor&) = default;
  DefaultedAttrsAccessor(DefaultedAttrsAccessor&&) = default;
  DefaultedAttrsAccessor(const AttrCollection& attrs, const AttrDefsAccessor& defs_accessor)
      : attrs_accesor_(attrs), defs_accessor_(defs_accessor) {}
  ~DefaultedAttrsAccessor() = default;

  bool HasAttrValue(const std::string& attr_name) const;
  Maybe<const AttrValue&> GetAttrValue(const std::string& attr_name) const;

  // Get values
  Maybe<bool> Bool(const std::string& name) const;
  Maybe<int64_t> Int64(const std::string& name) const;
  Maybe<double> Double(const std::string& name) const;
  Maybe<const std::string&> String(const std::string& name) const;
  Maybe<const PbRf<int64_t>&> ListInt64(const std::string& name) const;

 private:
  const AttrsAccessor& attrs_accessor() const { return attrs_accesor_; }
  const AttrDefsAccessor& defs_accessor() const { return defs_accessor_; }

  const AttrsAccessor attrs_accesor_;
  const AttrDefsAccessor defs_accessor_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_ATTR_H_
