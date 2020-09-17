#ifndef {{ util.module_header_macro_lock(module) }}
#define {{ util.module_header_macro_lock(module) }}

#include <memory>
#include <vector>
{% for dependency in util.module_dependencies(module) %}
#include "{{ util.module_cfg_header_name(dependency) }}"
{% endfor %}
#include "oneflow/cfg/repeated_field.h"
#include "oneflow/cfg/map_field.h"
#include "oneflow/cfg/shared_pair_iterator.h"

{% for package in util.module_package_list(module) %}
namespace {{ package }} {
{% endfor %}
namespace cfg {

{% for enm in util.module_enum_types(module) %}
enum {{ util.enum_name(enm) }} {
{% for value in util.enum_values(enm) %}
  {{ util.enum_value_name(value) }} = {{ util.enum_value_number(value) }},
{% endfor %}
};

inline ::std::string {{ util.enum_name(enm) }}_Name({{ util.enum_name(enm) }} value) {
  switch (value) {
{% for value in util.enum_values(enm) %}
  case {{ util.enum_value_name(value) }}: { return "{{ util.enum_value_name(value) }}"; }
{% endfor %}
  default:
    return "";
  }
}
{% endfor %}{# enm #}

{% for cls in util.module_message_types(module) %}
{% for field in util.message_type_fields(cls) %}
{# no duplicated class defined for each repeated field type #}
{% if util.field_has_repeated_label(field) and util.add_visited_repeated_field_type_name(field) %}

class {{ util.field_repeated_container_name(field) }};
// inheritance is helpful for avoid container iterator boilerplate 
class Const{{ util.field_repeated_container_name(field) }} : public ::oneflow::cfg::_RepeatedField_<{{ util.field_type_name(field) }}> {
 public:
  Const{{ util.field_repeated_container_name(field) }}(const ::std::shared_ptr<::std::vector<{{ util.field_type_name(field) }}>>& data): ::oneflow::cfg::_RepeatedField_<{{ util.field_type_name(field) }}>(data) {}
  Const{{ util.field_repeated_container_name(field) }}() = default;
  ~Const{{ util.field_repeated_container_name(field) }}() = default;

  bool operator==(const Const{{ util.field_repeated_container_name(field) }}& other) const {
    return *__SharedPtr__() == *other.__SharedPtr__();
  }
  bool operator<(const Const{{ util.field_repeated_container_name(field) }}& other) const {
    return *__SharedPtr__() < *other.__SharedPtr__();
  }
  // used by pybind11 only
  ::std::shared_ptr<Const{{ util.field_repeated_container_name(field) }}> __SharedConst__() const {
    return ::std::make_shared<Const{{ util.field_repeated_container_name(field) }}>(__SharedPtr__());
  }
{% if util.field_is_message_type(field) %}
  ::std::shared_ptr<Const{{ util.field_type_name(field) }}> __SharedConst__(::std::size_t index) const {
    return Get(index).__SharedConst__();
  }
{% endif %}{# message_type #}
};
class {{ util.field_repeated_container_name(field) }} final : public Const{{ util.field_repeated_container_name(field) }} {
 public:
  {{ util.field_repeated_container_name(field) }}(const ::std::shared_ptr<::std::vector<{{ util.field_type_name(field) }}>>& data): Const{{ util.field_repeated_container_name(field) }}(data) {}
  {{ util.field_repeated_container_name(field) }}() = default;
  ~{{ util.field_repeated_container_name(field) }}() = default;
  void CopyFrom(const Const{{ util.field_repeated_container_name(field) }}& other) {
    ::oneflow::cfg::_RepeatedField_<{{ util.field_type_name(field) }}>::CopyFrom(other);
  }
  void CopyFrom(const {{ util.field_repeated_container_name(field) }}& other) {
    ::oneflow::cfg::_RepeatedField_<{{ util.field_type_name(field) }}>::CopyFrom(other);
  }
  bool operator==(const {{ util.field_repeated_container_name(field) }}& other) const {
    return *__SharedPtr__() == *other.__SharedPtr__();
  }
  bool operator<(const {{ util.field_repeated_container_name(field) }}& other) const {
    return *__SharedPtr__() < *other.__SharedPtr__();
  }
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> __SharedMutable__() {
    return ::std::make_shared<{{ util.field_repeated_container_name(field) }}>(__SharedPtr__());
  }
{% if util.field_is_message_type(field) %}
  ::std::shared_ptr<{{ util.field_type_name(field) }}> __SharedAdd__() {
    return Add()->__SharedMutable__();
  }
  ::std::shared_ptr<{{ util.field_type_name(field) }}> __SharedMutable__(::std::size_t index) {
    return Mutable(index)->__SharedMutable__();
  }
{% endif %}{# message_type #}
};
{% endif  %}{# repeated #}
{# map begin #}
{% if util.field_is_map(field) and util.add_visited_map_field_type_name(field) %}

class {{ util.field_map_container_name(field) }};
// inheritance is helpful for avoid container iterator boilerplate 
class Const{{ util.field_map_container_name(field) }} : public ::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name(field) }}> {
 public:
  Const{{ util.field_map_container_name(field) }}(const ::std::shared_ptr<::std::map<{{ util.field_map_pair_type_name(field) }}>>& data): ::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name(field) }}>(data) {}
  Const{{ util.field_map_container_name(field) }}() = default;
  ~Const{{ util.field_map_container_name(field) }}() = default;

  bool operator==(const Const{{ util.field_map_container_name(field) }}& other) const {
    return *__SharedPtr__() == *other.__SharedPtr__();
  }
  bool operator<(const Const{{ util.field_map_container_name(field) }}& other) const {
    return *__SharedPtr__() < *other.__SharedPtr__();
  }
  // used by pybind11 only
  const {{ util.field_map_value_type_name(field) }}& Get(const {{ util.field_map_key_type_name(field) }}& key) const {
    return at(key);
  }

  // used by pybind11 only
  ::std::shared_ptr<Const{{ util.field_map_container_name(field) }}> __SharedConst__() const {
    return ::std::make_shared<Const{{ util.field_map_container_name(field) }}>(__SharedPtr__());
  }
{% if util.field_is_message_type(util.field_map_value_type(field)) %}
  // used by pybind11 only
  ::std::shared_ptr<Const{{ util.field_map_value_type_name(field) }}> __SharedConst__(const {{ util.field_map_key_type_name(field) }}& key) const {
    return at(key).__SharedConst__();
  }
  // used by pybind11 only
  using shared_const_iterator = ::oneflow::cfg::_SharedConstPairIterator_<Const{{ util.field_map_container_name(field) }}, Const{{ util.field_map_value_type_name(field) }}>;
  // ensuring mapped data's lifetime safety
  shared_const_iterator shared_const_begin() { return begin(); }
  // ensuring mapped data's lifetime safety
  shared_const_iterator shared_const_end() { return end(); }
{% endif %}{# message_type #}
};
class {{ util.field_map_container_name(field) }} final : public Const{{ util.field_map_container_name(field) }} {
 public:
  {{ util.field_map_container_name(field) }}(const ::std::shared_ptr<::std::map<{{ util.field_map_pair_type_name(field) }}>>& data): Const{{ util.field_map_container_name(field) }}(data) {}
  {{ util.field_map_container_name(field) }}() = default;
  ~{{ util.field_map_container_name(field) }}() = default;
  void CopyFrom(const Const{{ util.field_map_container_name(field) }}& other) {
    ::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name(field) }}>::CopyFrom(other);
  }
  void CopyFrom(const {{ util.field_map_container_name(field) }}& other) {
    ::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name(field) }}>::CopyFrom(other);
  }
  bool operator==(const {{ util.field_map_container_name(field) }}& other) const {
    return *__SharedPtr__() == *other.__SharedPtr__();
  }
  bool operator<(const {{ util.field_map_container_name(field) }}& other) const {
    return *__SharedPtr__() < *other.__SharedPtr__();
  }
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_map_container_name(field) }}> __SharedMutable__() {
    return ::std::make_shared<{{ util.field_map_container_name(field) }}>(__SharedPtr__());
  }

{% if util.field_is_message_type(util.field_map_value_type(field)) %}
  ::std::shared_ptr<{{ util.field_map_value_type_name(field) }}> __SharedMutable__(const {{ util.field_map_key_type_name(field) }}& key) {
    return (*this)[key].__SharedMutable__();
  }
  // used by pybind11 only
  using shared_mut_iterator = ::oneflow::cfg::_SharedMutPairIterator_<{{ util.field_map_container_name(field) }}, {{ util.field_map_value_type_name(field) }}>;
  // ensuring mapped data's lifetime safety
  shared_mut_iterator shared_mut_begin() { return begin(); }
  // ensuring mapped data's lifetime safety
  shared_mut_iterator shared_mut_end() { return end(); }
{% else %}
  void Set(const {{ util.field_map_key_type_name(field) }}& key, const {{ util.field_map_value_type_name(field) }}& value) {
    (*this)[key] = value;
  }
{% endif %}{# message_type #}
{# message_type #}
};
{% endif  %}{# map end #}
{% endfor %}{# field #}

class _{{ cls.name }}_ {
 public:
  _{{ cls.name }}_() { Clear(); }
  explicit _{{ cls.name }}_(const _{{ cls.name }}_& other) { CopyFrom(other); }
  explicit _{{ cls.name }}_(_{{ cls.name }}_&& other) = default;
  ~_{{ cls.name }}_() = default;
{% for oneof in util.message_type_oneofs(cls) %}

 // oneof enum {{ util.oneof_name(oneof) }}
 enum {{ util.oneof_enum_name(oneof) }} {
  {{ util.oneof_name(oneof).upper() }}_NOT_SET = 0,
  {% for field in util.oneof_type_fields(oneof) %}
  {{ util.oneof_type_field_enum_value_name(field) }} = {{ util.oneof_type_field_enum_value_number(field) }},
  {% endfor %}
 };
{% endfor %}{# oneof enum #}
  void Clear() {
{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
    clear_{{ util.field_name(field) }}();
{% elif util.field_has_repeated_label(field) or util.field_is_map(field) %}
    clear_{{ util.field_name(field) }}();
{% endif %}
{% endfor %}
{% for oneof in util.message_type_oneofs(cls) %}
    clear_{{util.oneof_name(oneof)}}();
{% endfor %}
  }
  void CopyFrom(const _{{ cls.name }}_& other) {
{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
    if (other.has_{{ util.field_name(field) }}()) {
{% if util.field_is_message_type(field) %}
      mutable_{{ util.field_name(field) }}()->CopyFrom(other.{{ util.field_name(field) }}());
{% else %}
      set_{{ util.field_name(field) }}(other.{{ util.field_name(field) }}());
{% endif %}
    } else {
      clear_{{ util.field_name(field) }}();
    }
{% elif util.field_has_repeated_label(field) or util.field_is_map(field) %}
    mutable_{{ util.field_name(field) }}()->CopyFrom(other.{{ util.field_name(field) }}());
{% endif %}
{% endfor %}
{% for oneof in util.message_type_oneofs(cls) %}
    {{util.oneof_name(oneof)}}_copy_from(other);
{% endfor %}{# oneofs #}
  }
{% for field in util.message_type_fields(cls) %}

{% if util.field_has_required_or_optional_label(field) %}
  // optional field {{ util.field_name(field) }}
 public:
{% if util.field_is_message_type(field) %}
  bool has_{{ util.field_name(field) }}() const {
    return has_{{ util.field_name(field) }}_;
  }
  const {{ util.field_type_name(field) }}& {{ util.field_name(field) }}() const {
    return {{ util.field_name(field) }}_;
  }
  void clear_{{ util.field_name(field) }}() {
    {{ util.field_name(field) }}_.Clear();
    has_{{ util.field_name(field) }}_ = false;
  }
  {{ util.field_type_name(field) }}* mutable_{{ util.field_name(field) }}() {
    has_{{ util.field_name(field) }}_ = true;
    return &{{ util.field_name(field) }}_;
  }
{% else %}
  bool has_{{ util.field_name(field) }}() const {
    return has_{{ util.field_name(field) }}_;
  }
  const {{ util.field_type_name(field) }}& {{ util.field_name(field) }}() const {
    if (has_{{ util.field_name(field) }}_) { return {{ util.field_name(field) }}_; }
{% if util.field_has_default_value(field) %}
    static const {{ util.field_type_name(field) }} default_static_value =
      {{ util.field_default_value_literal(field) }};
{% else %}
    static const {{ util.field_type_name(field) }} default_static_value = {{ util.field_type_name(field) }}();
{% endif %}
    return default_static_value;
  }
  void clear_{{ util.field_name(field) }}() {
    has_{{ util.field_name(field) }}_ = false;
  }
  void set_{{ util.field_name(field) }}(const {{ util.field_type_name(field) }}& value) {
    {{ util.field_name(field) }}_ = value;
    has_{{ util.field_name(field) }}_ = true;
  }
{% if util.field_type_name(field) == "::std::string" %}
  ::std::string* mutable_{{ util.field_name(field) }}() {
    has_{{ util.field_name(field) }}_ = true;
    return &{{ util.field_name(field) }}_;
  }
{% endif %}{# field string type #}
{% endif %}{# field_type #}
 protected:
  bool has_{{ util.field_name(field) }}_;
  {{ util.field_type_name(field) }} {{ util.field_name(field) }}_;
{% elif util.field_has_repeated_label(field) %}
  // repeated field {{ util.field_name(field) }}
 public:
  ::std::size_t {{ util.field_name(field) }}_size() const {
    return {{ util.field_name(field) }}_.size();
  }
  const {{ util.field_repeated_container_name(field) }}& {{ util.field_name(field) }}() const {
    return {{ util.field_name(field) }}_;
  }
  const {{ util.field_type_name(field) }}& {{ util.field_name(field) }}(::std::size_t index) const {
    return {{ util.field_name(field) }}_.Get(index);
  }
  void clear_{{ util.field_name(field) }}() {
    return {{ util.field_name(field) }}_.Clear();
  }
  {{ util.field_repeated_container_name(field) }}* mutable_{{ util.field_name(field) }}() {
    return  &{{ util.field_name(field) }}_;
  }
  {{ util.field_type_name(field) }}* mutable_{{ util.field_name(field) }}(::std::size_t index) {
    return  {{ util.field_name(field) }}_.Mutable(index);
  }
{% if util.field_is_message_type(field) %}
{% else %}
  void add_{{ util.field_name(field) }}(const {{ util.field_type_name(field) }}& value) {
    return {{ util.field_name(field) }}_.Add(value);
  }
{% endif %}{# field message type #}
 protected:
  {{ util.field_repeated_container_name(field) }} {{ util.field_name(field) }}_;
{% elif util.field_has_oneof_label(field) %}
 // oneof field {{ util.oneof_name_of_oneof_type_field(field) }}: {{ util.field_name(field) }}
 public:
  bool has_{{ util.field_name(field) }}() const {
    return {{ util.field_oneof_name(field) }}_case() == {{ util.oneof_type_field_enum_value_name(field) }};
  }
  void clear_{{ util.field_name(field) }}() {
    if (has_{{ util.field_name(field) }}()) {
{% if util.field_is_message_type(field) %}
      {{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_.Clear();
{% else %}
      {{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_ = {{ util.field_scalar_type_name(field) }}();
{% endif %}{# field message type #}
      {{ util.field_oneof_name(field) }}_case_ = {{ util.field_oneof_name(field).upper() }}_NOT_SET;
    }
  }
  const {{ util.field_type_name(field) }}& {{ util.field_name(field) }}() const {
    if (has_{{ util.field_name(field) }}()) {
      return {{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_;
    } else {
      static const {{ util.field_type_name(field) }} default_static_value = {{ util.field_type_name(field) }}();
      return default_static_value;
    }
  }
{% if util.field_is_message_type(field) %}
  {{ util.field_type_name(field) }}* mutable_{{ util.field_name(field) }}() {
    if(!has_{{ util.field_name(field) }}()) {
      clear_{{ util.field_oneof_name(field) }}();
    }
    {{ util.field_oneof_name(field) }}_case_ = {{ util.oneof_type_field_enum_value_name(field) }};
    return  &{{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_;
  }
{% else %}
  void set_{{ util.field_name(field) }}(const {{util.field_type_name(field) }}& value) {
    if(!has_{{ util.field_name(field) }}()) {
      clear_{{ util.field_oneof_name(field) }}();
    }
    {{ util.field_oneof_name(field) }}_case_ = {{ util.oneof_type_field_enum_value_name(field) }};
    {{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_ = value;
  }
{% if util.field_type_name(field) == "::std::string" %}
  ::std::string* mutable_{{ util.field_name(field) }}() {
    if(!has_{{ util.field_name(field) }}()) {
      clear_{{ util.field_oneof_name(field) }}();
    }
    {{ util.field_oneof_name(field) }}_case_ = {{ util.oneof_type_field_enum_value_name(field) }};
    return  &{{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_;
  }
{% endif %}{# field string type #}
{% endif %}{# field message type #}
{% elif util.field_is_map(field) %}
 public:
  ::std::size_t {{ util.field_name(field) }}_size() const {
    return {{ util.field_name(field) }}_.size();
  }
  const {{ util.field_map_container_name(field) }}& {{ util.field_name(field) }}() const {
    return {{ util.field_name(field) }}_;
  }

  {{ util.field_map_container_name(field) }} * mutable_{{ util.field_name(field) }}() {
    {{ util.field_map_container_name(field) }} * p = &{{ util.field_name(field) }}_;
    return p;
  }

  const {{ util.field_map_value_type_name(field) }}& {{ util.field_name(field) }}({{ util.field_map_key_type_name(field) }} key) const {
    return {{ util.field_name(field) }}_.at(key);
  }

  void clear_{{ util.field_name(field) }}() {
    return {{ util.field_name(field) }}_.Clear();
  }

{% if util.field_is_message_type(field) %}
{% else %}
  void add_{{ util.field_name(field) }}(const {{ util.field_type_name(field) }}& value) {
    return {{ util.field_name(field) }}_.Add(value);
  }
{% endif %}{# field message type #}
 protected:
  {{ util.field_map_container_name(field) }} {{ util.field_name(field) }}_;
{% endif %}{# label #}
{% endfor %}{# field #}
{% for oneof in util.message_type_oneofs(cls) %}
 
 public:
  // oneof {{ util.oneof_name(oneof) }}
  {{ util.oneof_enum_name(oneof) }} {{ util.oneof_name(oneof) }}_case() const {
    return {{ util.oneof_name(oneof) }}_case_;
  }
  bool has_{{util.oneof_name(oneof)}}() const {
    return {{ util.oneof_name(oneof) }}_case_ == {{ util.oneof_name(oneof).upper() }}_NOT_SET;
  }
 protected:
  void clear_{{util.oneof_name(oneof)}}() {
    switch ({{ util.oneof_name(oneof) }}_case()) {
{% for field in util.oneof_type_fields(oneof) %}
      case {{ util.oneof_type_field_enum_value_name(field) }}: {
{% if util.field_is_message_type(field) %}
        {{ util.oneof_name(oneof) }}_.{{ util.field_name(field) }}_.Clear();
{% else %}
        {{ util.oneof_name(oneof) }}_.{{ util.field_name(field) }}_ = {{ util.field_scalar_type_name(field) }}();
{% endif %}{# message_type #}
        break;
      }
{% endfor %}{# oneof_field #}
      case {{ util.oneof_name(oneof).upper() }}_NOT_SET: {
        break;
      }
    }
    {{ util.oneof_name(oneof) }}_case_ = {{ util.oneof_name(oneof).upper() }}_NOT_SET;
  }
  void {{util.oneof_name(oneof)}}_copy_from(const _{{ cls.name }}_& other) {
    switch (other.{{ util.oneof_name(oneof) }}_case()) {
{% for field in util.oneof_type_fields(oneof) %}
      case {{ util.oneof_type_field_enum_value_name(field) }}: {
{% if util.field_is_message_type(field) %}
        mutable_{{ util.field_name(field) }}()->CopyFrom(other.{{ util.field_name(field) }}());
{% else %}
        set_{{ util.field_name(field) }}(other.{{ util.field_name(field) }}());
{% endif %}{# message_type #}
        break;
      }
{% endfor %}{# oneof_field #}
      case {{ util.oneof_name(oneof).upper() }}_NOT_SET: {
        clear_{{util.oneof_name(oneof)}}();
      }
    }
  }
  struct {{ util.oneof_camel_name(oneof) }}Struct {
{% for field in util.oneof_type_fields(oneof) %}
{% if util.field_is_message_type(field) %}
   {{ util.field_message_type_name(field) }} {{ util.field_name(field) }}_;
{% else %}
   {{ util.field_scalar_type_name(field) }} {{ util.field_name(field) }}_;
{% endif %}{# field message type #}
{% endfor %}{# oneof_fields #}
  } {{ util.oneof_name(oneof) }}_;
  {{ util.oneof_enum_name(oneof) }} {{ util.oneof_name(oneof) }}_case_;
{% endfor %}{# message_oneof #}
 
 public:
  bool operator==(const _{{ cls.name }}_& other) const {
{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
    if (!(has_{{ util.field_name(field) }}() == other.has_{{ util.field_name(field) }}() && 
        {{ util.field_name(field) }}() == other.{{ util.field_name(field) }}())) {
      return false;
    }
{% elif util.field_has_repeated_label(field) or util.field_is_map(field) %}
    if (!({{ util.field_name(field) }}() == other.{{ util.field_name(field) }}())) {
      return false;
    }
{% endif %}{# field_label #}
{% endfor %}{# fields #}
{% for oneof in util.message_type_oneofs(cls) %}
    if (!({{ util.oneof_name(oneof) }}_case() == other.{{ util.oneof_name(oneof) }}_case())) {
      return false;
    }
    switch ({{ util.oneof_name(oneof) }}_case()) {
{% for field in util.oneof_type_fields(oneof) %}
      case {{ util.oneof_type_field_enum_value_name(field) }}: {
        if (!({{ util.field_name(field) }}() == other.{{ util.field_name(field) }}())) {
          return false;
        }
        break;
      }
{% endfor %}{# oneof_field #}
      case {{ util.oneof_name(oneof).upper() }}_NOT_SET: {
        break;
      }
    }
{% endfor %}{# oneofs #}
    return true;
  }

  bool operator<(const _{{ cls.name }}_& other) const {
{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
    if (!(has_{{ util.field_name(field) }}() == other.has_{{ util.field_name(field) }}())) {
      return has_{{ util.field_name(field) }}() < other.has_{{ util.field_name(field) }}();
    }
    if (!({{ util.field_name(field) }}() == other.{{ util.field_name(field) }}())) {
      return {{ util.field_name(field) }}() < other.{{ util.field_name(field) }}();
    }
{% elif util.field_has_repeated_label(field) or util.field_is_map(field) %}
    if (!({{ util.field_name(field) }}() == other.{{ util.field_name(field) }}())) {
      return {{ util.field_name(field) }}() < other.{{ util.field_name(field) }}();
    }
{% endif %}{# field_label #}
{% endfor %}{# fields #}
{% for oneof in util.message_type_oneofs(cls) %}
    if (!({{ util.oneof_name(oneof) }}_case() == other.{{ util.oneof_name(oneof) }}_case())) {
      return {{ util.oneof_name(oneof) }}_case() < other.{{ util.oneof_name(oneof) }}_case();
    }
    switch ({{ util.oneof_name(oneof) }}_case()) {
{% for field in util.oneof_type_fields(oneof) %}
      case {{ util.oneof_type_field_enum_value_name(field) }}: {
        if (!({{ util.field_name(field) }}() == other.{{ util.field_name(field) }}())) {
          return {{ util.field_name(field) }}() < other.{{ util.field_name(field) }}();
        }
        break;
      }
{% endfor %}{# oneof_field #}
      case {{ util.oneof_name(oneof).upper() }}_NOT_SET: {
        break;
      }
    }
{% endfor %}{# oneofs #}
    return false;
  }
};

class {{ cls.name }};
class Const{{ cls.name }} {
 public:
  Const{{ cls.name }}(const ::std::shared_ptr<::std::unique_ptr<_{{ cls.name }}_>>& data): data_(data) {}
  Const{{ cls.name }}(const Const{{ cls.name }}&) = default;
  Const{{ cls.name }}(Const{{ cls.name }}&&) noexcept = default;
  Const{{ cls.name }}(): data_(::std::make_shared<::std::unique_ptr<_{{ cls.name }}_>>()) {}
  ~Const{{ cls.name }}() = default;

  bool __Empty__() const {
    return !*data_;
  }
{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
  // required or optional field {{ util.field_name(field) }}
 public:
  bool has_{{ util.field_name(field) }}() const {
    return __SharedPtrOrDefault__()->has_{{ util.field_name(field) }}();
  }
  const {{ util.field_type_name(field) }}& {{ util.field_name(field) }}() const {
    return __SharedPtrOrDefault__()->{{ util.field_name(field) }}();
  }
  // used by pybind11 only
{% if util.field_is_message_type(field) %}
  ::std::shared_ptr<Const{{ util.field_type_name(field) }}> shared_const_{{ util.field_name(field) }}() const {
    return {{ util.field_name(field) }}().__SharedConst__();
  }
{% endif %}
{% elif util.field_has_repeated_label(field) %}
  // repeated field {{ util.field_name(field) }}
 public:
  ::std::size_t {{ util.field_name(field) }}_size() const {
    return __SharedPtrOrDefault__()->{{ util.field_name(field) }}_size();
  }
  const {{ util.field_repeated_container_name(field) }}& {{ util.field_name(field) }}() const {
    return __SharedPtrOrDefault__()->{{ util.field_name(field) }}();
  }
  const {{ util.field_type_name(field) }}& {{ util.field_name(field) }}(::std::size_t index) const {
    return __SharedPtrOrDefault__()->{{ util.field_name(field) }}(index);
  }
  // used by pybind11 only
  ::std::shared_ptr<Const{{ util.field_repeated_container_name(field) }}> shared_const_{{ util.field_name(field) }}() const {
    return {{ util.field_name(field) }}().__SharedConst__();
  }
{% if util.field_is_message_type(field) %}
  ::std::shared_ptr<Const{{ util.field_type_name(field) }}> shared_const_{{ util.field_name(field) }}(::std::size_t index) const {
    return {{ util.field_name(field) }}(index).__SharedConst__();
  }
{% else %}
{% endif %}{# field message type #}
{% elif util.field_has_oneof_label(field) %}
 // oneof field {{ util.oneof_name_of_oneof_type_field(field) }}: {{ util.field_name(field) }}
 public:
  bool has_{{ util.field_name(field) }}() const {
    return __SharedPtrOrDefault__()->has_{{ util.field_name(field) }}();
  }
  const {{ util.field_type_name(field) }}& {{ util.field_name(field) }}() const {
    return __SharedPtrOrDefault__()->{{ util.field_name(field) }}();
  }
  // used by pybind11 only
{% if util.field_is_message_type(field) %}
  ::std::shared_ptr<Const{{ util.field_type_name(field) }}> shared_const_{{ util.field_name(field) }}() const {
    return {{ util.field_name(field) }}().__SharedConst__();
  }
{% endif %}{# field message type #}
{# map begin#}
{% elif util.field_is_map(field) %}
  // map field {{ util.field_name(field) }}
 public:
  ::std::size_t {{ util.field_name(field) }}_size() const {
    return __SharedPtrOrDefault__()->{{ util.field_name(field) }}_size();
  }

  const {{ util.field_map_container_name(field) }}& {{ util.field_name(field) }}() const {
    return __SharedPtrOrDefault__()->{{ util.field_name(field) }}();
  }

  // used by pybind11 only
  ::std::shared_ptr<Const{{ util.field_map_container_name(field) }}> shared_const_{{ util.field_name(field) }}() const {
    return {{ util.field_name(field) }}().__SharedConst__();
  }
{# map end#}
{% endif %}{# field label type #}
{% endfor %}{# field #}
{% for oneof in util.message_type_oneofs(cls) %}
  using {{ util.oneof_enum_name(oneof) }} = _{{ cls.name }}_::{{ util.oneof_enum_name(oneof) }};
  {{ util.oneof_enum_name(oneof) }} {{ util.oneof_name(oneof) }}_case() const {
    return __SharedPtrOrDefault__()->{{ util.oneof_name(oneof) }}_case();
  }
{% endfor %}{# oneofs #}

  ::std::shared_ptr<Const{{ cls.name }}> __SharedConst__() const {
    return ::std::make_shared<Const{{ cls.name }}>(data_);
  }
  int64_t __Id__() const { return reinterpret_cast<int64_t>(data_.get()); }
  // the data of `this` will be moved to the result which is mutable
  ::std::shared_ptr<{{ cls.name }}> __Move__();
 public:
  bool operator==(const Const{{ cls.name }}& other) const {
    return *__SharedPtrOrDefault__() == *other.__SharedPtrOrDefault__();
  }

  bool operator<(const Const{{ cls.name }}& other) const {
    return *__SharedPtrOrDefault__() < *other.__SharedPtrOrDefault__();
  }
 protected:
  const ::std::unique_ptr<_{{ cls.name }}_>& __SharedPtrOrDefault__() const {
    if (*data_) { return *data_; }
    static const ::std::unique_ptr<_{{ cls.name }}_> default_ptr(new _{{ cls.name }}_());
    return default_ptr;
  }
  const ::std::unique_ptr<_{{ cls.name }}_>& __SharedPtr__() {
    return *__SharedUniquePtr__();
  }
  const ::std::shared_ptr<::std::unique_ptr<_{{ cls.name }}_>>& __SharedUniquePtr__() {
    if (!*data_) { data_->reset(new _{{ cls.name }}_()); }
    return data_;
  }
  // use std::shared_ptr for sharing reference between mutable object and const object
  // use std::unique_ptr for moving ownership 
  ::std::shared_ptr<::std::unique_ptr<_{{ cls.name }}_>> data_;
};

class {{ cls.name }} final : public Const{{ cls.name }} {
 public:
  {{ cls.name }}(const ::std::shared_ptr<::std::unique_ptr<_{{ cls.name }}_>>& data)
    : Const{{ cls.name }}(data) {}
  {{ cls.name }}(const {{ cls.name }}& other) { CopyFrom(other); }
  // enable nothrow for std::vector<{{ cls.name }}> resize 
  {{ cls.name }}({{ cls.name }}&&) noexcept = default;
  {{ cls.name }}() = default;
  ~{{ cls.name }}() = default;

  bool operator==(const {{ cls.name }}& other) const {
    return *__SharedPtrOrDefault__() == *other.__SharedPtrOrDefault__();
  }

  bool operator<(const {{ cls.name }}& other) const {
    return *__SharedPtrOrDefault__() < *other.__SharedPtrOrDefault__();
  }
  void Clear() {
    if (data_) { data_->reset(); }
  }
  void CopyFrom(const {{ cls.name }}& other) {
    if (other.__Empty__()) {
      Clear();
    } else {
      __SharedPtr__()->CopyFrom(**other.data_);
    }
  }
  {{ cls.name }}& operator=(const {{ cls.name }}& other) {
    CopyFrom(other);
    return *this;
  }

{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
  // required or optional field {{ util.field_name(field) }}
 public:
  void clear_{{ util.field_name(field) }}() {
    return __SharedPtr__()->clear_{{ util.field_name(field) }}();
  }
{% if util.field_is_message_type(field) %}
  {{ util.field_type_name(field) }}* mutable_{{ util.field_name(field) }}() {
    return __SharedPtr__()->mutable_{{ util.field_name(field) }}();
  }
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_type_name(field) }}> shared_mutable_{{ util.field_name(field) }}() {
    return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
  }
{% else %}
  void set_{{ util.field_name(field) }}(const {{ util.field_type_name(field) }}& value) {
    return __SharedPtr__()->set_{{ util.field_name(field) }}(value);
  }
{% if util.field_type_name(field) == "::std::string" %}
  ::std::string* mutable_{{ util.field_name(field) }}() {
    return  __SharedPtr__()->mutable_{{ util.field_name(field) }}();
  }
{% endif %}{# field string type #}
{% endif %}
{% elif util.field_has_repeated_label(field) %}
  // repeated field {{ util.field_name(field) }}
 public:
  void clear_{{ util.field_name(field) }}() {
    return __SharedPtr__()->clear_{{ util.field_name(field) }}();
  }
  {{ util.field_repeated_container_name(field) }}* mutable_{{ util.field_name(field) }}() {
    return __SharedPtr__()->mutable_{{ util.field_name(field) }}();
  }
  {{ util.field_type_name(field) }}* mutable_{{ util.field_name(field) }}(::std::size_t index) {
    return __SharedPtr__()->mutable_{{ util.field_name(field) }}(index);
  }
{% if util.field_is_message_type(field) %}
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> shared_mutable_{{ util.field_name(field) }}() {
    return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
  }
  ::std::shared_ptr<{{ util.field_type_name(field) }}> shared_mutable_{{ util.field_name(field) }}(::std::size_t index) {
    return mutable_{{ util.field_name(field) }}(index)->__SharedMutable__();
  }
{% else %}
  void add_{{ util.field_name(field) }}(const {{ util.field_type_name(field) }}& value) {
    return __SharedPtr__()->add_{{ util.field_name(field) }}(value);
  }
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> shared_mutable_{{ util.field_name(field) }}() {
    return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
  }
{% endif %}{# field message type #}
{% elif util.field_has_oneof_label(field) %}
  void clear_{{ util.field_name(field) }}() {
    return __SharedPtr__()->clear_{{ util.field_name(field) }}();
  }
{% if util.field_is_message_type(field) %}
  {{ util.field_type_name(field) }}* mutable_{{ util.field_name(field) }}() {
    return __SharedPtr__()->mutable_{{ util.field_name(field) }}();
  }
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_type_name(field) }}> shared_mutable_{{ util.field_name(field) }}() {
    return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
  }
{% else %}
  void set_{{ util.field_name(field) }}(const {{ util.field_type_name(field) }}& value) {
    return __SharedPtr__()->set_{{ util.field_name(field) }}(value);
  }
{% if util.field_type_name(field) == "::std::string" %}
  ::std::string* mutable_{{ util.field_name(field) }}() {
    return  __SharedPtr__()->mutable_{{ util.field_name(field) }}();
  }
{% endif %}{# field string type #}
{% endif %}{# field message type #}
{# map begin#}
{% elif util.field_is_map(field) %}
  // repeated field {{ util.field_name(field) }}
 public:
  void clear_{{ util.field_name(field) }}() {
    return __SharedPtr__()->clear_{{ util.field_name(field) }}();
  }

  const {{ util.field_map_container_name(field) }} & {{ util.field_name(field) }}() {
    return __SharedPtr__()->{{ util.field_name(field) }}();
  }

  {{ util.field_map_container_name(field) }}* mutable_{{ util.field_name(field) }}() {
    return __SharedPtr__()->mutable_{{ util.field_name(field) }}();
  }

  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_map_container_name(field) }}> shared_mutable_{{ util.field_name(field) }}() {
    return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
  }
{# map end#}
{% endif %}{# field label type #}
{% endfor %}{# field #}

  ::std::shared_ptr<{{ cls.name }}> __SharedMutable__() {
    return ::std::make_shared<{{ cls.name }}>(__SharedUniquePtr__());
  }
};

inline ::std::shared_ptr<{{ cls.name }}> Const{{ cls.name }}::__Move__() {
  if (__Empty__()) { return ::std::make_shared<{{ cls.name }}>(); }
  auto data = std::make_shared<::std::unique_ptr<_{{ cls.name }}_>>();
  *data = std::move(*data_);
  return ::std::make_shared<{{ cls.name }}>(data);
}

{% endfor %}{# cls #}
}
{% for package in util.module_package_list(module) %}
}
{% endfor %}{# package #}
#endif  // {{ util.module_header_macro_lock(module) }}
