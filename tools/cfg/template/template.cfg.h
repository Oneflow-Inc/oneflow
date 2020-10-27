#ifndef {{ util.module_header_macro_lock(module) }}
#define {{ util.module_header_macro_lock(module) }}

#include <memory>
#include <vector>
#include <map>
{% for dependency in util.module_dependencies(module) %}
#include "{{ util.module_cfg_header_name(dependency) }}"
{% endfor %}
#include "oneflow/cfg/repeated_field.h"
#include "oneflow/cfg/map_field.h"
#include "oneflow/cfg/message.h"
#include "oneflow/cfg/shared_pair_iterator.h"
#include "{{ util.module_proto_header_name(module) }}"

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

inline {{ util.module_package_namespace(module) }}::{{ util.enum_name(enm) }} Cfg{{ util.enum_name(enm) }}ToProto{{ util.enum_name(enm) }}(const {{ util.module_package_namespace(module) }}::cfg::{{ util.enum_name(enm) }}& cfg_enum) {
  return {{ util.module_package_namespace(module) }}::{{ util.enum_name(enm) }}(int(cfg_enum));
}

inline {{ util.module_package_namespace(module) }}::cfg::{{ util.enum_name(enm) }} Proto{{ util.enum_name(enm) }}ToCfg{{ util.enum_name(enm) }}(const {{ util.module_package_namespace(module) }}::{{ util.enum_name(enm) }}& proto_enum) {
  return {{ util.module_package_namespace(module) }}::cfg::{{ util.enum_name(enm) }}(int(proto_enum));
}
{% endfor %}{# enm #}

{% for cls in util.module_nested_message_types(module) %}
{% if not util.class_is_map_entry(cls) %}
class {{ util.class_name(cls) }};
class Const{{ util.class_name(cls) }} : public ::oneflow::cfg::Message {
 public:
{% for oneof in util.message_type_oneofs(cls) %}

 // oneof enum {{ util.oneof_name(oneof) }}
 enum {{ util.oneof_enum_name(oneof) }} {
  {{ util.oneof_name(oneof).upper() }}_NOT_SET = 0,
  {% for field in util.oneof_type_fields(oneof) %}
  {{ util.oneof_type_field_enum_value_name(field) }} = {{ util.field_number(field) }},
  {% endfor %}
 };
{% endfor %}{# oneof enum #}

  class _{{ util.class_name(cls) }}_ {
   public:
    _{{ util.class_name(cls) }}_() { Clear(); }
    explicit _{{ util.class_name(cls) }}_(const _{{ util.class_name(cls) }}_& other) { CopyFrom(other); }
    explicit _{{ util.class_name(cls) }}_(_{{ util.class_name(cls) }}_&& other) = default;
    _{{ util.class_name(cls) }}_(const {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}& proto_{{ util.class_name(cls).lower() }}) {
  {% for field in util.message_type_fields(cls) %}
  {% if util.field_has_required_or_optional_label(field) %}
      // required_or_optional field: {{ util.field_name(field) }}
      if (proto_{{ util.class_name(cls).lower() }}.has_{{ util.field_name(field) }}()) {
  {% if util.field_is_message_type(field)%}
        *mutable_{{ util.field_name(field) }}() = {{ util.field_message_type_name_with_cfg_namespace(field) }}(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}());
  {% elif util.field_is_enum_type(field) %}
        set_{{ util.field_name(field) }}(Proto{{ util.field_enum_name(field) }}ToCfg{{ util.field_enum_name(field) }}(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()));
  {% else %}
        set_{{ util.field_name(field) }}(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}());
  {% endif %}{# message_type #}
      }
  {% elif util.field_has_repeated_label(field) %}
      // repeated field: {{ util.field_name(field) }}
      if (!proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}().empty()) {
  {% if util.field_is_message_type(field)%}
        for (const {{ util.field_message_type_name_with_proto_namespace(field) }}& elem : proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}() ) {
          *mutable_{{ util.field_name(field) }}()->Add() = {{ util.field_message_type_name_with_cfg_namespace(field) }}(elem);
        }
  {% elif util.field_is_enum_type(field) %}
        for (const int& elem : proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}() ) {
          add_{{ util.field_name(field) }}({{ util.module_package_namespace(module) }}::cfg::{{ util.field_enum_name(field) }}(elem));
        }
  {% else %}
        for (const {{ util.field_type_name_with_cfg_namespace(field) }}& elem : proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()) {
          add_{{ util.field_name(field) }}(elem);
        }
  {% endif %}{# field_type #}
      }
  {% elif util.field_has_map_label(field) %}
      // map field : {{ util.field_name(field) }}
      if (!proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}().empty()) {
        ::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>&  mut_{{ util.field_name(field) }} = *mutable_{{ util.field_name(field) }}();
        for (const auto& pair : proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()) {
  {% if util.field_map_value_type_is_message(field) %}
          mut_{{ util.field_name(field) }}[pair.first] = {{ util.field_map_value_type_name_with_cfg_namespace(field) }}(pair.second);
  {% elif util.field_map_value_type_is_enum(field) %}
          mut_{{ util.field_name(field) }}[pair.first] = Proto{{ util.field_map_value_type_enum_name(field) }}ToCfg{{ util.field_map_value_type_enum_name(field) }}(pair.second);
  {% else %}
          mut_{{ util.field_name(field) }}[pair.first] = pair.second;
  {% endif %}{# map_value_type #}
        }
      }
  {% endif %}{# field_label #}
  {% endfor %}{# field #}
  {% for oneof in util.message_type_oneofs(cls) %}
      // oneof field: {{ util.oneof_name(oneof) }}
      {{ util.oneof_enum_name(oneof) }} {{ util.oneof_name(oneof) }}_case = {{ util.oneof_enum_name(oneof) }}(int(proto_{{ util.class_name(cls).lower() }}.{{ util.oneof_name(oneof) }}_case()));
      switch ({{ util.oneof_name(oneof) }}_case) {
  {% for field in util.oneof_type_fields(oneof) %}
        case {{ util.oneof_type_field_enum_value_name(field) }}: {
  {% if util.field_is_message_type(field) %}
          *mutable_{{ util.field_name(field) }}() = {{ util.field_message_type_name_with_cfg_namespace(field) }}(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}());
  {% elif util.field_is_enum_type(field) %}
          set_{{ util.field_name(field) }}(Proto{{ util.field_enum_name(field) }}ToCfg{{ util.field_enum_name(field) }}(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()));
  {% else %}
          set_{{ util.field_name(field) }}(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}());
  {% endif %}{# message_type #}
          break;
        }
  {% endfor %}{# oneof_field #}
        case {{ util.oneof_name(oneof).upper() }}_NOT_SET: {
          break;
        }
      }
  {% endfor %}{# oneofs #}
    }
    ~_{{ util.class_name(cls) }}_() = default;

    void ToProto({{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}* proto_{{ util.class_name(cls).lower() }}) const {
  {% for field in util.message_type_fields(cls) %}
  {% if util.field_has_required_or_optional_label(field) %}
      // required_or_optional field: {{ util.field_name(field) }}
      if (this->has_{{ util.field_name(field) }}()) {
  {% if util.field_is_message_type(field)%}
        {{ util.field_message_type_name_with_proto_namespace(field) }} proto_{{ util.field_name(field).lower() }};
        {{ util.field_name(field) }}().ToProto(&proto_{{ util.field_name(field).lower() }});
        proto_{{ util.class_name(cls).lower() }}->mutable_{{ util.field_name(field) }}()->CopyFrom(proto_{{ util.field_name(field).lower() }});
  {% elif util.field_is_enum_type(field) %}
        proto_{{ util.class_name(cls).lower() }}->set_{{ util.field_name(field) }}(Cfg{{ util.field_enum_name(field) }}ToProto{{ util.field_enum_name(field) }}({{ util.field_name(field) }}()));
  {% else %}
        proto_{{ util.class_name(cls).lower() }}->set_{{ util.field_name(field) }}({{ util.field_name(field) }}());
  {% endif %}{# message_type #}
        }
  {% elif util.field_has_repeated_label(field) %}
      // repeated field: {{ util.field_name(field) }}
      if (!{{ util.field_name(field) }}().empty()) {
  {% if util.field_is_message_type(field)%}
        for (const {{ util.field_message_type_name_with_cfg_namespace(field) }}& elem : {{ util.field_name(field) }}() ) {
          {{ util.field_message_type_name_with_proto_namespace(field) }} proto_{{ util.field_name(field).lower() }}_elem;
          elem.ToProto(&proto_{{ util.field_name(field).lower() }}_elem);
          *proto_{{ util.class_name(cls).lower() }}->mutable_{{ util.field_name(field) }}()->Add() = proto_{{ util.field_name(field).lower() }}_elem;
        }
  {% elif util.field_is_enum_type(field) %}
        for (const int& elem : {{ util.field_name(field) }}() ) {
          proto_{{ util.class_name(cls).lower() }}->add_{{ util.field_name(field) }}(::oneflow::{{ util.field_enum_name(field) }}(elem));
        }
  {% else %}
        for (const {{ util.field_type_name_with_cfg_namespace(field) }}& elem : {{ util.field_name(field) }}()) {
          proto_{{ util.class_name(cls).lower() }}->add_{{ util.field_name(field) }}(elem);
        }
  {% endif %}{# message_type #}
      }
  {% elif util.field_has_map_label(field) %}
      // map field : {{ util.field_name(field) }}
      if (!{{ util.field_name(field) }}().empty()) {
        auto& mut_{{ util.field_name(field) }} = *(proto_{{ util.class_name(cls).lower() }}->mutable_{{ util.field_name(field) }}());
        for (const auto& pair : {{ util.field_name(field) }}()) {
  {% if util.field_map_value_type_is_message(field) %}
          {{ util.module_package_namespace(module) }}::{{ util.field_map_value_type_name(field) }} proto_{{ util.field_name(field).lower() }}_value;
          pair.second.ToProto(&proto_{{ util.field_name(field).lower() }}_value);
          mut_{{ util.field_name(field) }}[pair.first] = proto_{{ util.field_name(field).lower() }}_value;
  {% elif util.field_map_value_type_is_enum(field) %}
          mut_{{ util.field_name(field) }}[pair.first] = Cfg{{ util.field_map_value_type_enum_name(field) }}ToProto{{ util.field_map_value_type_enum_name(field) }}(pair.second);
  {% else %}
          mut_{{ util.field_name(field) }}[pair.first] = pair.second;
  {% endif %}{# map_value_type #}
        }
      }
  {% endif %}{# field_type #}
  {% endfor %}{# field #}

  {% for oneof in util.message_type_oneofs(cls) %}
    // oneof field: {{ util.oneof_name(oneof) }}
      {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.oneof_enum_name(oneof) }} {{ util.oneof_name(oneof) }}_case = {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.oneof_enum_name(oneof) }}(int(this->{{ util.oneof_name(oneof) }}_case()));
      switch ({{ util.oneof_name(oneof) }}_case) {
  {% for field in util.oneof_type_fields(oneof) %}
        case {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.oneof_type_field_enum_value_name(field) }}: {
  {% if util.field_is_message_type(field) %}
          {{ util.field_message_type_name_with_proto_namespace(field) }} of_proto_{{ util.field_name(field).lower() }};
          {{ util.field_name(field) }}().ToProto(&of_proto_{{ util.field_name(field).lower() }});
          proto_{{ util.class_name(cls).lower() }}->mutable_{{ util.field_name(field) }}()->CopyFrom(of_proto_{{ util.field_name(field).lower() }});
  {% elif util.field_is_enum_type(field) %}
          proto_{{ util.class_name(cls).lower() }}->set_{{ util.field_name(field) }}(Cfg{{ util.field_enum_name(field) }}ToProto{{ util.field_enum_name(field) }}({{ util.field_name(field) }}()));
  {% else %}
          proto_{{ util.class_name(cls).lower() }}->set_{{ util.field_name(field) }}({{ util.field_name(field) }}());
  {% endif %}{# message_type #}
          break;
        }
  {% endfor %}{# oneof_field #}
        case {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.oneof_name(oneof).upper() }}_NOT_SET: {
          break;
        }
      }
  {% endfor %}{# oneofs #}
    }

    ::std::string DebugString() const {
      {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }} proto_{{ util.class_name(cls).lower() }};
      this->ToProto(&proto_{{ util.class_name(cls).lower() }});
      return proto_{{ util.class_name(cls).lower() }}.DebugString();
    }

    void Clear() {
  {% for field in util.message_type_fields(cls) %}
  {% if util.field_has_required_or_optional_label(field) %}
      clear_{{ util.field_name(field) }}();
  {% elif util.field_has_repeated_label(field) or util.field_has_map_label(field) %}
      clear_{{ util.field_name(field) }}();
  {% endif %}
  {% endfor %}
  {% for oneof in util.message_type_oneofs(cls) %}
      clear_{{util.oneof_name(oneof)}}();
  {% endfor %}
    }
    void CopyFrom(const _{{ util.class_name(cls) }}_& other) {
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
  {% elif util.field_has_repeated_label(field) or util.field_has_map_label(field) %}
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
    const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}() const {
      return {{ util.field_name(field) }}_;
    }
    void clear_{{ util.field_name(field) }}() {
      {{ util.field_name(field) }}_.Clear();
      has_{{ util.field_name(field) }}_ = false;
    }
    {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}() {
      has_{{ util.field_name(field) }}_ = true;
      return &{{ util.field_name(field) }}_;
    }
  {% else %}
    bool has_{{ util.field_name(field) }}() const {
      return has_{{ util.field_name(field) }}_;
    }
    const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}() const {
      if (has_{{ util.field_name(field) }}_) { return {{ util.field_name(field) }}_; }
  {% if util.field_has_default_value(field) %}
      static const {{ util.field_type_name_with_cfg_namespace(field) }} default_static_value =
        {{ util.field_default_value_literal(field) }};
  {% else %}
      static const {{ util.field_type_name_with_cfg_namespace(field) }} default_static_value = {{ util.field_type_name_with_cfg_namespace(field) }}();
  {% endif %}
      return default_static_value;
    }
    void clear_{{ util.field_name(field) }}() {
      has_{{ util.field_name(field) }}_ = false;
    }
    void set_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value) {
      {{ util.field_name(field) }}_ = value;
      has_{{ util.field_name(field) }}_ = true;
    }
    {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}() {
      has_{{ util.field_name(field) }}_ = true;
      return &{{ util.field_name(field) }}_;
    }
  {% endif %}{# field_type #}
   protected:
    bool has_{{ util.field_name(field) }}_;
    {{ util.field_type_name_with_cfg_namespace(field) }} {{ util.field_name(field) }}_;
  {% elif util.field_has_repeated_label(field) %}
    // repeated field {{ util.field_name(field) }}
   public:
    ::std::size_t {{ util.field_name(field) }}_size() const {
      return {{ util.field_name(field) }}_.size();
    }
    const ::oneflow::cfg::_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>& {{ util.field_name(field) }}() const {
      return {{ util.field_name(field) }}_;
    }
    const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}(::std::size_t index) const {
      return {{ util.field_name(field) }}_.Get(index);
    }
    void clear_{{ util.field_name(field) }}() {
      return {{ util.field_name(field) }}_.Clear();
    }
    ::oneflow::cfg::_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>* mutable_{{ util.field_name(field) }}() {
      return  &{{ util.field_name(field) }}_;
    }
    {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}(::std::size_t index) {
      return  {{ util.field_name(field) }}_.Mutable(index);
    }
  {% if util.field_is_message_type(field) %}
    {{ util.field_type_name_with_cfg_namespace(field) }}* add_{{ util.field_name(field) }}() {
      return {{ util.field_name(field) }}_.Add();
    }
  {% else %}
    void add_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value) {
      return {{ util.field_name(field) }}_.Add(value);
    }
  {% endif %}{# field message type #}
   protected:
    ::oneflow::cfg::_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}> {{ util.field_name(field) }}_;
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
    const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}() const {
      if (has_{{ util.field_name(field) }}()) {
        return {{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_;
      } else {
        static const {{ util.field_type_name_with_cfg_namespace(field) }} default_static_value = {{ util.field_type_name_with_cfg_namespace(field) }}();
        return default_static_value;
      }
    }
  {% if util.field_is_message_type(field) %}
    {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}() {
      if(!has_{{ util.field_name(field) }}()) {
        clear_{{ util.field_oneof_name(field) }}();
      }
      {{ util.field_oneof_name(field) }}_case_ = {{ util.oneof_type_field_enum_value_name(field) }};
      return  &{{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_;
    }
  {% else %}
    void set_{{ util.field_name(field) }}(const {{util.field_type_name_with_cfg_namespace(field) }}& value) {
      if(!has_{{ util.field_name(field) }}()) {
        clear_{{ util.field_oneof_name(field) }}();
      }
      {{ util.field_oneof_name(field) }}_case_ = {{ util.oneof_type_field_enum_value_name(field) }};
      {{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_ = value;
    }
    {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}() {
      if(!has_{{ util.field_name(field) }}()) {
        clear_{{ util.field_oneof_name(field) }}();
      }
      {{ util.field_oneof_name(field) }}_case_ = {{ util.oneof_type_field_enum_value_name(field) }};
      return  &{{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_;
    }
  {% endif %}{# field message type #}
  {% elif util.field_has_map_label(field) %}
   public:
    ::std::size_t {{ util.field_name(field) }}_size() const {
      return {{ util.field_name(field) }}_.size();
    }
    const ::oneflow::cfg::_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>& {{ util.field_name(field) }}() const {
      return {{ util.field_name(field) }}_;
    }

    ::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}> * mutable_{{ util.field_name(field) }}() {
      ::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>
       * p = &{{ util.field_name(field) }}_;
      return p;
    }

    const {{ util.field_map_value_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}({{ util.field_map_key_type_name(field) }} key) const {
      return {{ util.field_name(field) }}_.at(key);
    }

    void clear_{{ util.field_name(field) }}() {
      return {{ util.field_name(field) }}_.Clear();
    }

  {% if util.field_is_message_type(field) %}
  {% else %}
    void add_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value) {
      return {{ util.field_name(field) }}_.Add(value);
    }
  {% endif %}{# field message type #}
   protected:
    ::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}> {{ util.field_name(field) }}_;
  {% endif %}{# label #}
  {% endfor %}{# field #}
  {% for oneof in util.message_type_oneofs(cls) %}

   public:
    // oneof {{ util.oneof_name(oneof) }}
    {{ util.oneof_enum_name(oneof) }} {{ util.oneof_name(oneof) }}_case() const {
      return {{ util.oneof_name(oneof) }}_case_;
    }
    bool has_{{util.oneof_name(oneof)}}() const {
      return {{ util.oneof_name(oneof) }}_case_ != {{ util.oneof_name(oneof).upper() }}_NOT_SET;
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
    void {{util.oneof_name(oneof)}}_copy_from(const _{{ util.class_name(cls) }}_& other) {
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
    int compare(const _{{ util.class_name(cls) }}_& other) {
  {% for field in util.message_type_fields(cls) %}
  {% if util.field_has_required_or_optional_label(field) %}
      if (!(has_{{ util.field_name(field) }}() == other.has_{{ util.field_name(field) }}())) {
        return has_{{ util.field_name(field) }}() < other.has_{{ util.field_name(field) }}() ? -1 : 1;
      } else if (!({{ util.field_name(field) }}() == other.{{ util.field_name(field) }}())) {
        return {{ util.field_name(field) }}() < other.{{ util.field_name(field) }}() ? -1 : 1;
      }
  {% elif util.field_has_repeated_label(field) or util.field_has_map_label(field) %}
      if (!({{ util.field_name(field) }}() == other.{{ util.field_name(field) }}())) {
        return {{ util.field_name(field) }}() < other.{{ util.field_name(field) }}() ? -1 : 1;
      }
  {% endif %}{# field_label #}
  {% endfor %}{# fields #}
  {% for oneof in util.message_type_oneofs(cls) %}
      if (!({{ util.oneof_name(oneof) }}_case() == other.{{ util.oneof_name(oneof) }}_case())) {
        return {{ util.oneof_name(oneof) }}_case() < other.{{ util.oneof_name(oneof) }}_case() ? -1 : 1;
      }
      switch ({{ util.oneof_name(oneof) }}_case()) {
  {% for field in util.oneof_type_fields(oneof) %}
        case {{ util.oneof_type_field_enum_value_name(field) }}: {
          if (!({{ util.field_name(field) }}() == other.{{ util.field_name(field) }}())) {
            return {{ util.field_name(field) }}() < other.{{ util.field_name(field) }}() ? -1 : 1;
          }
          break;
        }
  {% endfor %}{# oneof_field #}
        case {{ util.oneof_name(oneof).upper() }}_NOT_SET: {
          break;
        }
      }
  {% endfor %}{# oneofs #}
      return 0;
    }

    bool operator==(const _{{ util.class_name(cls) }}_& other) const {
  {% for field in util.message_type_fields(cls) %}
  {% if util.field_has_required_or_optional_label(field) %}
      if (!(has_{{ util.field_name(field) }}() == other.has_{{ util.field_name(field) }}() &&
          {{ util.field_name(field) }}() == other.{{ util.field_name(field) }}())) {
        return false;
      }
  {% elif util.field_has_repeated_label(field) or util.field_has_map_label(field) %}
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

    bool operator<(const _{{ util.class_name(cls) }}_& other) const {
  {% for field in util.message_type_fields(cls) %}
  {% if util.field_has_required_or_optional_label(field) %}
      if (!(has_{{ util.field_name(field) }}() == other.has_{{ util.field_name(field) }}())) {
        return has_{{ util.field_name(field) }}() < other.has_{{ util.field_name(field) }}();
      }
      if (!({{ util.field_name(field) }}() == other.{{ util.field_name(field) }}())) {
        return {{ util.field_name(field) }}() < other.{{ util.field_name(field) }}();
      }
  {% elif util.field_has_repeated_label(field) or util.field_has_map_label(field) %}
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

  Const{{ util.class_name(cls) }}(const ::std::shared_ptr<::std::unique_ptr<_{{ util.class_name(cls) }}_>>& data): data_(data) {}
  Const{{ util.class_name(cls) }}(const Const{{ util.class_name(cls) }}&) = default;
  Const{{ util.class_name(cls) }}(Const{{ util.class_name(cls) }}&&) noexcept = default;
  Const{{ util.class_name(cls) }}(): data_(::std::make_shared<::std::unique_ptr<_{{ util.class_name(cls) }}_>>()) {}
  Const{{ util.class_name(cls) }}(const {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}& proto_{{ util.class_name(cls).lower() }}) {
    data_ = ::std::make_shared<::std::unique_ptr<_{{ util.class_name(cls) }}_>>(new _{{ util.class_name(cls) }}_(proto_{{ util.class_name(cls).lower() }}));
  }
  ~Const{{ util.class_name(cls) }}() override = default;

  void ToProto({{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}* proto_{{ util.class_name(cls).lower() }}) const {
    __SharedPtrOrDefault__()->ToProto(proto_{{ util.class_name(cls).lower() }});
  }

  ::std::string DebugString() const {
    return __SharedPtrOrDefault__()->DebugString();
  }

  bool __Empty__() const {
    return !*data_;
  }

  int FieldNumber4FieldName(const std::string& field_name) const override {
    static const ::std::map<std::string, int> field_name2field_number{
{% for field in util.message_type_fields(cls) %}
      {"{{ util.field_name(field) }}", {{ util.field_number(field) }}},
{% endfor %}{# field #}
    };
    const auto& iter = field_name2field_number.find(field_name);
    if (iter != field_name2field_number.end()) { return iter->second; }
    return 0;
  }

  bool FieldDefined4FieldNumber(int field_number) const override {
    switch (field_number) {
{% for field in util.message_type_fields(cls) %}
      case {{ util.field_number(field) }}:
{% endfor %}{# field #}
        return true;
      default:
        return false;
    }
  }

  const std::set<std::type_index>& ValidTypeIndices4FieldNumber(int field_number) const {
    switch (field_number) {
{% for field in util.message_type_fields(cls) %}
      case {{ util.field_number(field) }}: {
        static const ::std::set<::std::type_index> type_indices{
{% if util.field_has_repeated_label(field) %}
          typeid(::oneflow::cfg::_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>),
          typeid(::oneflow::cfg::_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>),
{% elif util.field_has_map_label(field) %}
          typeid(::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>),
          typeid(::oneflow::cfg::_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>),
{% else %}
          typeid({{ util.field_type_name(field) }}),
{% if util.field_is_message_type(field) %}
          typeid(::oneflow::cfg::Message),
          typeid(Const{{ util.field_type_name(field) }}),
{% endif %}{# field message type #}
{% endif %}{# field_label #}
        };
        return type_indices;
      }
{% endfor %}{# field #}
      default: {
        static const ::std::set<::std::type_index> empty;
        return empty;
      }
    }
  }

  const void* FieldPtr4FieldNumber(int field_number) const override {
    switch (field_number) {
{% for field in util.message_type_fields(cls) %}
      case {{ util.field_number(field) }}: return &{{ util.field_name(field) }}();
{% endfor %}{# field #}
      default: return nullptr;
    }
  }

{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
  // required or optional field {{ util.field_name(field) }}
 public:
  bool has_{{ util.field_name(field) }}() const {
    return __SharedPtrOrDefault__()->has_{{ util.field_name(field) }}();
  }
  const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}() const {
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
  const ::oneflow::cfg::_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>& {{ util.field_name(field) }}() const {
    return __SharedPtrOrDefault__()->{{ util.field_name(field) }}();
  }
  const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}(::std::size_t index) const {
    return __SharedPtrOrDefault__()->{{ util.field_name(field) }}(index);
  }
  // used by pybind11 only
  ::std::shared_ptr<::oneflow::cfg::_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>> shared_const_{{ util.field_name(field) }}() const {
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
  const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}() const {
    return __SharedPtrOrDefault__()->{{ util.field_name(field) }}();
  }
  // used by pybind11 only
{% if util.field_is_message_type(field) %}
  ::std::shared_ptr<Const{{ util.field_type_name(field) }}> shared_const_{{ util.field_name(field) }}() const {
    return {{ util.field_name(field) }}().__SharedConst__();
  }
{% endif %}{# field message type #}
{# map begin#}
{% elif util.field_has_map_label(field) %}
  // map field {{ util.field_name(field) }}
 public:
  ::std::size_t {{ util.field_name(field) }}_size() const {
    return __SharedPtrOrDefault__()->{{ util.field_name(field) }}_size();
  }

  const ::oneflow::cfg::_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>& {{ util.field_name(field) }}() const {
    return __SharedPtrOrDefault__()->{{ util.field_name(field) }}();
  }

  // used by pybind11 only
  ::std::shared_ptr<::oneflow::cfg::_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>> shared_const_{{ util.field_name(field) }}() const {
    return {{ util.field_name(field) }}().__SharedConst__();
  }
{# map end#}
{% endif %}{# field label type #}
{% endfor %}{# field #}
{% for oneof in util.message_type_oneofs(cls) %}
  {{ util.oneof_enum_name(oneof) }} {{ util.oneof_name(oneof) }}_case() const {
    return __SharedPtrOrDefault__()->{{ util.oneof_name(oneof) }}_case();
  }

  bool has_{{ util.oneof_name(oneof) }}() const {
    return __SharedPtrOrDefault__()->has_{{ util.oneof_name(oneof) }}();
  }
{% endfor %}{# oneofs #}

  ::std::shared_ptr<Const{{ util.class_name(cls) }}> __SharedConst__() const {
    return ::std::make_shared<Const{{ util.class_name(cls) }}>(data_);
  }
  int64_t __Id__() const { return reinterpret_cast<int64_t>(data_.get()); }
  // the data of `this` will be moved to the result which is mutable
  ::std::shared_ptr<{{ util.class_name(cls) }}> __Move__();
 public:
  bool operator==(const Const{{ util.class_name(cls) }}& other) const {
    return *__SharedPtrOrDefault__() == *other.__SharedPtrOrDefault__();
  }

  bool operator<(const Const{{ util.class_name(cls) }}& other) const {
    return *__SharedPtrOrDefault__() < *other.__SharedPtrOrDefault__();
  }
 protected:
  const ::std::unique_ptr<_{{ util.class_name(cls) }}_>& __SharedPtrOrDefault__() const {
    if (*data_) { return *data_; }
    static const ::std::unique_ptr<_{{ util.class_name(cls) }}_> default_ptr(new _{{ util.class_name(cls) }}_());
    return default_ptr;
  }
  const ::std::unique_ptr<_{{ util.class_name(cls) }}_>& __SharedPtr__() {
    return *__SharedUniquePtr__();
  }
  const ::std::shared_ptr<::std::unique_ptr<_{{ util.class_name(cls) }}_>>& __SharedUniquePtr__() {
    if (!*data_) { data_->reset(new _{{ util.class_name(cls) }}_()); }
    return data_;
  }
  // use std::shared_ptr for sharing reference between mutable object and const object
  // use std::unique_ptr for moving ownership
  ::std::shared_ptr<::std::unique_ptr<_{{ util.class_name(cls) }}_>> data_;
};

class {{ util.class_name(cls) }} final : public Const{{ util.class_name(cls) }} {
 public:
  {{ util.class_name(cls) }}(const ::std::shared_ptr<::std::unique_ptr<_{{ util.class_name(cls) }}_>>& data)
    : Const{{ util.class_name(cls) }}(data) {}
  {{ util.class_name(cls) }}(const {{ util.class_name(cls) }}& other) { CopyFrom(other); }
  // enable nothrow for std::vector<{{ util.class_name(cls) }}> resize
  {{ util.class_name(cls) }}({{ util.class_name(cls) }}&&) noexcept = default;
  {{ util.class_name(cls) }}() = default;
    {{ util.class_name(cls) }}(const {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}& proto_{{ util.class_name(cls).lower() }})
    : Const{{ util.class_name(cls) }}(proto_{{ util.class_name(cls).lower() }}) {}

  ~{{ util.class_name(cls) }}() = default;

  void* MutableFieldPtr4FieldNumber(int field_number) override {
    switch (field_number) {
{% for field in util.message_type_fields(cls) %}
      case {{ util.field_number(field) }}: return mutable_{{ util.field_name(field) }}();
{% endfor %}{# field #}
      default: return nullptr;
    }
  }


  bool operator==(const {{ util.class_name(cls) }}& other) const {
    return *__SharedPtrOrDefault__() == *other.__SharedPtrOrDefault__();
  }

  bool operator<(const {{ util.class_name(cls) }}& other) const {
    return *__SharedPtrOrDefault__() < *other.__SharedPtrOrDefault__();
  }
  void Clear() {
    if (data_) { data_->reset(); }
  }
  void CopyFrom(const {{ util.class_name(cls) }}& other) {
    if (other.__Empty__()) {
      Clear();
    } else {
      __SharedPtr__()->CopyFrom(**other.data_);
    }
  }
  {{ util.class_name(cls) }}& operator=(const {{ util.class_name(cls) }}& other) {
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
  {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}() {
    return __SharedPtr__()->mutable_{{ util.field_name(field) }}();
  }
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> shared_mutable_{{ util.field_name(field) }}() {
    return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
  }
{% else %}
  void set_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value) {
    return __SharedPtr__()->set_{{ util.field_name(field) }}(value);
  }
  {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}() {
    return  __SharedPtr__()->mutable_{{ util.field_name(field) }}();
  }
{% endif %}
{% elif util.field_has_repeated_label(field) %}
  // repeated field {{ util.field_name(field) }}
 public:
  void clear_{{ util.field_name(field) }}() {
    return __SharedPtr__()->clear_{{ util.field_name(field) }}();
  }
  ::oneflow::cfg::_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>* mutable_{{ util.field_name(field) }}() {
    return __SharedPtr__()->mutable_{{ util.field_name(field) }}();
  }
  {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}(::std::size_t index) {
    return __SharedPtr__()->mutable_{{ util.field_name(field) }}(index);
  }
{% if util.field_is_message_type(field) %}
  // used by pybind11 only
  ::std::shared_ptr<::oneflow::cfg::_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>> shared_mutable_{{ util.field_name(field) }}() {
    return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
  }
  ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> shared_mutable_{{ util.field_name(field) }}(::std::size_t index) {
    return mutable_{{ util.field_name(field) }}(index)->__SharedMutable__();
  }
  {{ util.field_type_name_with_cfg_namespace(field) }}* add_{{ util.field_name(field) }}() {
    return __SharedPtr__()->add_{{ util.field_name(field) }}();
  }
{% else %}
  void add_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value) {
    return __SharedPtr__()->add_{{ util.field_name(field) }}(value);
  }
  // used by pybind11 only
  ::std::shared_ptr<::oneflow::cfg::_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>> shared_mutable_{{ util.field_name(field) }}() {
    return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
  }
{% endif %}{# field message type #}
{% elif util.field_has_oneof_label(field) %}
  void clear_{{ util.field_name(field) }}() {
    return __SharedPtr__()->clear_{{ util.field_name(field) }}();
  }
{% if util.field_is_message_type(field) %}
  {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}() {
    return __SharedPtr__()->mutable_{{ util.field_name(field) }}();
  }
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> shared_mutable_{{ util.field_name(field) }}() {
    return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
  }
{% else %}
  void set_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value) {
    return __SharedPtr__()->set_{{ util.field_name(field) }}(value);
  }
  {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}() {
    return  __SharedPtr__()->mutable_{{ util.field_name(field) }}();
  }
{% endif %}{# field message type #}
{# map begin#}
{% elif util.field_has_map_label(field) %}
  // repeated field {{ util.field_name(field) }}
 public:
  void clear_{{ util.field_name(field) }}() {
    return __SharedPtr__()->clear_{{ util.field_name(field) }}();
  }

  const ::oneflow::cfg::_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}> & {{ util.field_name(field) }}() {
    return __SharedPtr__()->{{ util.field_name(field) }}();
  }

  ::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>* mutable_{{ util.field_name(field) }}() {
    return __SharedPtr__()->mutable_{{ util.field_name(field) }}();
  }

  // used by pybind11 only
  ::std::shared_ptr<::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}> > shared_mutable_{{ util.field_name(field) }}() {
    return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
  }
{# map end#}
{% endif %}{# field label type #}
{% endfor %}{# field #}

  ::std::shared_ptr<{{ util.class_name(cls) }}> __SharedMutable__() {
    return ::std::make_shared<{{ util.class_name(cls) }}>(__SharedUniquePtr__());
  }
};

inline ::std::shared_ptr<{{ util.class_name(cls) }}> Const{{ util.class_name(cls) }}::__Move__() {
  if (__Empty__()) { return ::std::make_shared<{{ util.class_name(cls) }}>(); }
  auto data = std::make_shared<::std::unique_ptr<_{{ util.class_name(cls) }}_>>();
  *data = std::move(*data_);
  return ::std::make_shared<{{ util.class_name(cls) }}>(data);
}
{% endif %}{# cls is not entry #}
{% endfor %}{# cls #}
}
{% for package in util.module_package_list(module) %}
} // namespace {{ package }}
{% endfor %}{# package #}
#endif  // {{ util.module_header_macro_lock(module) }}
