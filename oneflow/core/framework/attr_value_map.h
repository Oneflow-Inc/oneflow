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
#ifndef ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_MAP_H_
#define ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_MAP_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/user_op_attr.cfg.h"

namespace oneflow {

class AttrValueMap {
 public:
  AttrValueMap()
      : attrs_(new HashMap<std::string, std::shared_ptr<cfg::AttrValue>>{}),
        attr_names_(new HashSet<std::string>{}) {}
  virtual ~AttrValueMap() = default;

  size_t size() const { return attr_names_->size(); }
  bool empty() const { return attr_names_->empty(); }

  bool has_parents() const { return parents_ && !parents_->empty(); }
  Maybe<std::vector<AttrValueMap>> parents() const { return parents_; }

  Maybe<bool> HasAttr(const std::string& attr_name) const {
    CHECK_OR_RETURN(attr_names_);
    return attr_names_->count(attr_name);
  }

  template<typename T>
  Maybe<T> GetAttr(const std::string& attr_name) const;

  static Maybe<AttrValueMap> Compose(const AttrValueMap& attrs, const AttrValueMap& parent);

  class iterator {
   public:
    using NameSetIter = HashSet<std::string>::const_iterator;
    using AttrValueMapIter = HashMap<std::string, std::shared_ptr<cfg::AttrValue>>::const_iterator;
    using const_reference = HashMap<std::string, std::shared_ptr<cfg::AttrValue>>::const_reference;
    using const_pointer = HashMap<std::string, std::shared_ptr<cfg::AttrValue>>::const_pointer;

    iterator() = default;
    explicit iterator(const AttrValueMap* self, const NameSetIter& it, const NameSetIter& end_it)
        : self_(self), name_set_it_(it), name_set_end_it_(end_it) {
      UpdateAttrValueMapIter();
    }

    iterator& operator++() {
      ++name_set_it_;
      UpdateAttrValueMapIter();
      return *this;
    }
    explicit operator bool() const { return name_set_it_ != name_set_end_it_; }

    const_reference operator*() const { return attr_value_map_it_.operator*(); }
    const_pointer operator->() const { return attr_value_map_it_.operator->(); }

    bool operator==(const iterator& other) const {
      return self_ == other.self_ && name_set_it_ == other.name_set_it_;
    }
    bool operator!=(const iterator& other) const {
      return self_ != other.self_ || name_set_it_ != other.name_set_it_;
    }

    AttrValueMapIter internal() const { return attr_value_map_it_; }

   private:
    void UpdateAttrValueMapIter() {
      if (name_set_it_ != name_set_end_it_) { attr_value_map_it_ = self_->Find(*name_set_it_); }
    }

   private:
    const AttrValueMap* self_;
    NameSetIter name_set_it_;
    NameSetIter name_set_end_it_;

    AttrValueMapIter attr_value_map_it_;
  };

  friend class iterator;

  iterator begin() const { return iterator(this, attr_names_->begin(), attr_names_->end()); }
  iterator end() const { return iterator(this, attr_names_->end(), attr_names_->end()); }

  iterator find(const std::string& attr_name) const {
    const auto& name_set_it = attr_names_->find(attr_name);
    return iterator(this, name_set_it, attr_names_->end());
  }

 protected:
  friend class MutableAttrValueMap;
  iterator::AttrValueMapIter Find(const std::string& attr_name) const;

  std::shared_ptr<std::vector<AttrValueMap>> parents_;
  std::shared_ptr<HashMap<std::string, std::shared_ptr<cfg::AttrValue>>> attrs_;
  std::shared_ptr<HashSet<std::string>> attr_names_;
};

class MutableAttrValueMap : public AttrValueMap {
 public:
  MutableAttrValueMap() : AttrValueMap() {}
  MutableAttrValueMap(const MutableAttrValueMap&) = delete;
  virtual ~MutableAttrValueMap() = default;

  MutableAttrValueMap& operator=(const MutableAttrValueMap&) = delete;

  template<typename T>
  Maybe<void> SetAttr(const std::string& attr_name, const T& attr_val);

  Maybe<MutableAttrValueMap&> Compose(const AttrValueMap& parent);
  static Maybe<MutableAttrValueMap> Compose(const AttrValueMap& attrs, const AttrValueMap& parent);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_MAP_H_
