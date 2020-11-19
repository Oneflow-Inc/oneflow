#include "{{ util.module_cfg_header_name(module) }}"
{% for dependency in util.module_dependencies(module) %}
#include "{{ util.module_cfg_header_name(dependency) }}"
{% endfor %}
#include "{{ util.module_proto_header_name(module) }}"

{% for package in util.module_package_list(module) %}
namespace {{ package }} {
{% endfor %}
namespace cfg {
using PbMessage = ::google::protobuf::Message;

{% for cls in util.module_nested_message_types(module) %}
{% if not util.class_is_map_entry(cls) %}
Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::_{{ util.class_name(cls) }}_() { Clear(); }
Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::_{{ util.class_name(cls) }}_(const _{{ util.class_name(cls) }}_& other) { CopyFrom(other); }
Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::_{{ util.class_name(cls) }}_(const {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}& proto_{{ util.class_name(cls).lower() }}) {
  InitFromProto(proto_{{ util.class_name(cls).lower() }});
}
Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::_{{ util.class_name(cls) }}_(_{{ util.class_name(cls) }}_&& other) = default;
Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::~_{{ util.class_name(cls) }}_() = default;

void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::InitFromProto(const {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}& proto_{{ util.class_name(cls).lower() }}) {
{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
  // required_or_optional field: {{ util.field_name(field) }}
  if (proto_{{ util.class_name(cls).lower() }}.has_{{ util.field_name(field) }}()) {
{%if util.field_is_message_type(field)%}
  *mutable_{{ util.field_name(field) }}() = {{ util.field_message_type_name_with_cfg_namespace(field) }}(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}());      
{% elif util.field_is_enum_type(field) %}
  set_{{ util.field_name(field) }}(static_cast<std::remove_reference<std::remove_const<decltype({{ util.field_name(field) }}())>::type>::type>(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()));
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
      add_{{ util.field_name(field) }}({{ util.module_package_cfg_namespace(module) }}::{{ util.field_enum_name(field) }}(elem));
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
{{ util.field_map_container_name(field) }}&  mut_{{ util.field_name(field) }} = *mutable_{{ util.field_name(field) }}();
    for (const auto& pair : proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()) {
{% if util.field_map_value_type_is_message(field) %}
      mut_{{ util.field_name(field) }}[pair.first] = {{ util.field_map_value_type_name_with_cfg_namespace(field) }}(pair.second);
{% elif util.field_map_value_type_is_enum(field) %}
      mut_{{ util.field_name(field) }}[pair.first] = static_cast<std::remove_const<std::remove_reference<decltype(mut_{{ util.field_name(field) }}[pair.first])>::type>::type>(pair.second);
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
      set_{{ util.field_name(field) }}(static_cast<std::remove_const<std::remove_reference<decltype({{ util.field_name(field) }}())>::type>::type>(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()));
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

void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::ToProto({{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}* proto_{{ util.class_name(cls).lower() }}) const {
{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
  // required_or_optional field: {{ util.field_name(field) }}
  if (this->has_{{ util.field_name(field) }}()) {
{% if util.field_is_message_type(field)%}
    {{ util.field_message_type_name_with_proto_namespace(field) }} proto_{{ util.field_name(field).lower() }};
    {{ util.field_name(field) }}().ToProto(&proto_{{ util.field_name(field).lower() }});
    proto_{{ util.class_name(cls).lower() }}->mutable_{{ util.field_name(field) }}()->CopyFrom(proto_{{ util.field_name(field).lower() }});
{% elif util.field_is_enum_type(field) %}
    proto_{{ util.class_name(cls).lower() }}->set_{{ util.field_name(field) }}(static_cast<std::remove_const<std::remove_reference<decltype(proto_{{ util.class_name(cls).lower() }}->{{ util.field_name(field) }}())>::type>::type>({{ util.field_name(field) }}()));
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
      proto_{{ util.class_name(cls).lower() }}->add_{{ util.field_name(field) }}({{ util.module_package_namespace(module) }}::{{ util.field_enum_name(field) }}(elem));
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
      mut_{{ util.field_name(field) }}[pair.first] = static_cast<std::remove_const<std::remove_reference<decltype(mut_{{ util.field_name(field) }}[pair.first])>::type>::type>(pair.second);
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
      proto_{{ util.class_name(cls).lower() }}->set_{{ util.field_name(field) }}(static_cast<decltype(proto_{{ util.class_name(cls).lower() }}->{{ util.field_name(field) }}())>({{ util.field_name(field) }}()));
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

::std::string Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::DebugString() const {
  {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }} proto_{{ util.class_name(cls).lower() }};
  this->ToProto(&proto_{{ util.class_name(cls).lower() }});
  return proto_{{ util.class_name(cls).lower() }}.DebugString();
}

void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::Clear() {
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

void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::CopyFrom(const _{{ util.class_name(cls) }}_& other) {
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
{% if util.field_is_message_type(field) %}
bool Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::has_{{ util.field_name(field) }}() const {
  return has_{{ util.field_name(field) }}_;
}
const {{ util.field_type_name_with_cfg_namespace(field) }}& Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::{{ util.field_name(field) }}() const {
  if (!{{ util.field_name(field) }}_) {
    static const ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> default_static_value =
      ::std::make_shared<{{ util.field_type_name_with_cfg_namespace(field) }}>();
    return *default_static_value;
  }
  return *({{ util.field_name(field) }}_.get());
}
void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::clear_{{ util.field_name(field) }}() {
  if ({{ util.field_name(field) }}_) {
    {{ util.field_name(field) }}_->Clear();
  }
  has_{{ util.field_name(field) }}_ = false;
}
{{ util.field_type_name_with_cfg_namespace(field) }}* Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::mutable_{{ util.field_name(field) }}() {
  if (!{{ util.field_name(field) }}_) {
    {{ util.field_name(field) }}_ = ::std::make_shared<{{ util.field_type_name_with_cfg_namespace(field) }}>();
  }
  has_{{ util.field_name(field) }}_ = true;
  return {{ util.field_name(field) }}_.get();
}
{% else %}
bool Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::has_{{ util.field_name(field) }}() const {
  return has_{{ util.field_name(field) }}_;
}
const {{ util.field_type_name_with_cfg_namespace(field) }}& Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::{{ util.field_name(field) }}() const {
  if (has_{{ util.field_name(field) }}_) { return {{ util.field_name(field) }}_; }
{% if util.field_has_default_value(field) %}
  static const {{ util.field_type_name_with_cfg_namespace(field) }} default_static_value =
    {{ util.field_type_name_with_cfg_namespace(field) }}({{ util.field_default_value_literal(field) }});
{% else %}
  static const {{ util.field_type_name_with_cfg_namespace(field) }} default_static_value = {{ util.field_type_name_with_cfg_namespace(field) }}();
{% endif %}
  return default_static_value;
}
void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::clear_{{ util.field_name(field) }}() {
  has_{{ util.field_name(field) }}_ = false;
}
void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::set_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value) {
  {{ util.field_name(field) }}_ = value;
  has_{{ util.field_name(field) }}_ = true;
}
{{ util.field_type_name_with_cfg_namespace(field) }}* Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::mutable_{{ util.field_name(field) }}() {
  has_{{ util.field_name(field) }}_ = true;
  return &{{ util.field_name(field) }}_;
}
{% endif %}{# field_type #}
{% elif util.field_has_repeated_label(field) %}
// repeated field {{ util.field_name(field) }}
::std::size_t Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::{{ util.field_name(field) }}_size() const {
  if (!{{ util.field_name(field) }}_) {
    static const ::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> default_static_value =
      ::std::make_shared<{{ util.field_repeated_container_name(field) }}>();
    return default_static_value->size();
  }
  return {{ util.field_name(field) }}_->size();
}
const {{ util.field_repeated_container_name(field) }}& Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::{{ util.field_name(field) }}() const {
  if (!{{ util.field_name(field) }}_) {
    static const ::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> default_static_value =
      ::std::make_shared<{{ util.field_repeated_container_name(field) }}>();
    return *(default_static_value.get());
  }
  return *({{ util.field_name(field) }}_.get());
}
const {{ util.field_type_name_with_cfg_namespace(field) }}& Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::{{ util.field_name(field) }}(::std::size_t index) const {
  if (!{{ util.field_name(field) }}_) {
    static const ::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> default_static_value =
      ::std::make_shared<{{ util.field_repeated_container_name(field) }}>();
    return default_static_value->Get(index);
  }
  return {{ util.field_name(field) }}_->Get(index);
}
void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::clear_{{ util.field_name(field) }}() {
  if (!{{ util.field_name(field) }}_) {
    {{ util.field_name(field) }}_ = ::std::make_shared<{{ util.field_repeated_container_name(field) }}>();
  }
  return {{ util.field_name(field) }}_->Clear();
}
{{ util.field_repeated_container_name(field) }}* Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::mutable_{{ util.field_name(field) }}() {
  if (!{{ util.field_name(field) }}_) {
    {{ util.field_name(field) }}_ = ::std::make_shared<{{ util.field_repeated_container_name(field) }}>();
  }
  return  {{ util.field_name(field) }}_.get();
}
{{ util.field_type_name_with_cfg_namespace(field) }}* Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::mutable_{{ util.field_name(field) }}(::std::size_t index) {
  if (!{{ util.field_name(field) }}_) {
    {{ util.field_name(field) }}_ = ::std::make_shared<{{ util.field_repeated_container_name(field) }}>();
  }
  return  {{ util.field_name(field) }}_->Mutable(index);
}
{% if util.field_is_message_type(field) %}
{{ util.field_type_name_with_cfg_namespace(field) }}* Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::add_{{ util.field_name(field) }}() {
  if (!{{ util.field_name(field) }}_) {
    {{ util.field_name(field) }}_ = ::std::make_shared<{{ util.field_repeated_container_name(field) }}>();
  }
  return {{ util.field_name(field) }}_->Add();
}
{% else %}
void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::add_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value) {
  if (!{{ util.field_name(field) }}_) {
    {{ util.field_name(field) }}_ = ::std::make_shared<{{ util.field_repeated_container_name(field) }}>();
  }
  return {{ util.field_name(field) }}_->Add(value);
}
{% endif %}{# field message type #}
{% elif util.field_has_oneof_label(field) %}
// oneof field {{ util.oneof_name_of_oneof_type_field(field) }}: {{ util.field_name(field) }}
bool Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::has_{{ util.field_name(field) }}() const {
  return {{ util.field_oneof_name(field) }}_case() == {{ util.oneof_type_field_enum_value_name(field) }};
}
void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::clear_{{ util.field_name(field) }}() {
  if (has_{{ util.field_name(field) }}()) {
{% if util.field_is_message_type(field) %}
    {
      using Shared_ptr = ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}>;
      Shared_ptr* __attribute__((__may_alias__)) ptr = reinterpret_cast<Shared_ptr*>(&({{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_));
      ptr->~Shared_ptr();
    }
{% elif util.field_is_string_type(field) %}
    {
      using String = ::std::string;
      String* ptr = reinterpret_cast<String*>(&({{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_)[0]);
      ptr->~String();
    }
{% else %}
    {{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_ = {{ util.field_scalar_type_name(field) }}();
{% endif %}{# field message type #}
    {{ util.field_oneof_name(field) }}_case_ = {{ util.field_oneof_name(field).upper() }}_NOT_SET;
  }
}

const {{ util.field_type_name_with_cfg_namespace(field) }}& Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::{{ util.field_name(field) }}() const {
  if (has_{{ util.field_name(field) }}()) {
  {% if util.field_is_message_type(field) %}
    const ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}>* __attribute__((__may_alias__)) ptr = reinterpret_cast<const ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}>*>(&({{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_));
    return *(*ptr);
  {% elif util.field_is_string_type(field) %}
    const ::std::string* ptr = reinterpret_cast<const ::std::string*>(&({{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_)[0]);
    return *ptr;
  {% else %}
    return {{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_;
  {% endif %}
  } else {
  {% if util.field_is_message_type(field) %}
    static const ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> default_static_value = ::std::make_shared<{{ util.field_type_name_with_cfg_namespace(field) }}>();
    return *default_static_value;
  {% else %}
    static const {{ util.field_type_name_with_cfg_namespace(field) }} default_static_value = {{ util.field_type_name_with_cfg_namespace(field) }}();
    return default_static_value;
  {% endif %}
  }
}
{% if util.field_is_message_type(field) %}
{{ util.field_type_name_with_cfg_namespace(field) }}* Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::mutable_{{ util.field_name(field) }}() {
  if(!has_{{ util.field_name(field) }}()) {
    clear_{{ util.field_oneof_name(field) }}();
    new (&({{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_)) ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}>(new {{ util.field_type_name_with_cfg_namespace(field) }}());
  }
  {{ util.field_oneof_name(field) }}_case_ = {{ util.oneof_type_field_enum_value_name(field) }};
  ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}>* __attribute__((__may_alias__)) ptr = reinterpret_cast<::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}>*>(&({{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_));
  return  (*ptr).get();
}
{% else %}
void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::set_{{ util.field_name(field) }}(const {{util.field_type_name_with_cfg_namespace(field) }}& value) {
  if(!has_{{ util.field_name(field) }}()) {
    clear_{{ util.field_oneof_name(field) }}();
  {% if util.field_is_string_type(field) %}
    new (&({{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_)) std::string();
  {% endif %}
  }
  {{ util.field_oneof_name(field) }}_case_ = {{ util.oneof_type_field_enum_value_name(field) }};
  {% if util.field_is_string_type(field) %}
  std::string* ptr = reinterpret_cast<std::string*>(&({{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_)[0]);
  *ptr = value;
  {% else %}
  {{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_ = value;
  {% endif %}
}
{{ util.field_type_name_with_cfg_namespace(field) }}* Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::mutable_{{ util.field_name(field) }}() {
  if(!has_{{ util.field_name(field) }}()) {
    clear_{{ util.field_oneof_name(field) }}();
  {% if util.field_is_string_type(field) %}
    new (&({{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_)) std::string();
  {% endif %}
  }
  {% if util.field_is_string_type(field) %}
  ::std::string* ptr = reinterpret_cast<::std::string*>(&({{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_)[0]);
  return ptr;
  {% else %}
  {{ util.field_oneof_name(field) }}_case_ = {{ util.oneof_type_field_enum_value_name(field) }};
  return  &{{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_;
  {% endif %}
}
{% endif %}{# field message type #}
{% elif util.field_has_map_label(field) %}
::std::size_t Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::{{ util.field_name(field) }}_size() const {
  if (!{{ util.field_name(field) }}_) {
    static const ::std::shared_ptr<{{ util.field_map_container_name(field) }}> default_static_value =
      ::std::make_shared<{{ util.field_map_container_name(field) }}>();
    return default_static_value->size();
  }
  return {{ util.field_name(field) }}_->size();
}
const {{ util.field_map_container_name(field) }}& Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::{{ util.field_name(field) }}() const {
  if (!{{ util.field_name(field) }}_) {
    static const ::std::shared_ptr<{{ util.field_map_container_name(field) }}> default_static_value =
      ::std::make_shared<{{ util.field_map_container_name(field) }}>();
    return *(default_static_value.get());
  }
  return *({{ util.field_name(field) }}_.get());
}

{{ util.field_map_container_name(field) }} * Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::mutable_{{ util.field_name(field) }}() {
  if (!{{ util.field_name(field) }}_) {
    {{ util.field_name(field) }}_ = ::std::make_shared<{{ util.field_map_container_name(field) }}>();
  }
  return {{ util.field_name(field) }}_.get();
}

const {{ util.field_map_value_type_name_with_cfg_namespace(field) }}& Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::{{ util.field_name(field) }}({{ util.field_map_key_type_name(field) }} key) const {
  if (!{{ util.field_name(field) }}_) {
    static const ::std::shared_ptr<{{ util.field_map_container_name(field) }}> default_static_value =
      ::std::make_shared<{{ util.field_map_container_name(field) }}>();
    return default_static_value->at(key);
  }
  return {{ util.field_name(field) }}_->at(key);
}

void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::clear_{{ util.field_name(field) }}() {
  if (!{{ util.field_name(field) }}_) {
    {{ util.field_name(field) }}_ = ::std::make_shared<{{ util.field_map_container_name(field) }}>();
  }
  return {{ util.field_name(field) }}_->Clear();
}

{% if util.field_is_message_type(field) %}
{% else %}
void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::add_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value) {
  if (!{{ util.field_name(field) }}_) {
    {{ util.field_name(field) }}_ = ::std::make_shared<{{ util.field_map_container_name(field) }}>();
  }
  return {{ util.field_name(field) }}_->Add(value);
}
{% endif %}{# field message type #}
{% endif %}{# label #}
{% endfor %}{# field #}
{% for oneof in util.message_type_oneofs(cls) %}
Const{{ util.class_name(cls) }}::{{ util.oneof_enum_name(oneof) }} Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::{{ util.oneof_name(oneof) }}_case() const {
  return {{ util.oneof_name(oneof) }}_case_;
}
bool Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::has_{{util.oneof_name(oneof)}}() const {
  return {{ util.oneof_name(oneof) }}_case_ != {{ util.oneof_name(oneof).upper() }}_NOT_SET;
}
void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::clear_{{util.oneof_name(oneof)}}() {
  switch ({{ util.oneof_name(oneof) }}_case()) {
{% for field in util.oneof_type_fields(oneof) %}
    case {{ util.oneof_type_field_enum_value_name(field) }}: {
{% if util.field_is_message_type(field) %}
      {
        using Shared_ptr = ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}>;
        Shared_ptr* __attribute__((__may_alias__)) ptr = reinterpret_cast<Shared_ptr*>(&({{ util.field_oneof_name(field) }}_.{{ util.field_name(field) }}_));
        ptr->~Shared_ptr();
      }
{% elif util.field_is_string_type(field) %}
      {
        using String = ::std::string;
        String* ptr = reinterpret_cast<String*>(&({{ util.oneof_name(oneof) }}_.{{ util.field_name(field) }}_)[0]);
        ptr->~String();
      }
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
void Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::{{util.oneof_name(oneof)}}_copy_from(const _{{ util.class_name(cls) }}_& other) {
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
{% endfor %}{# message_oneof #}


int Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::compare(const _{{ util.class_name(cls) }}_& other) {
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

bool Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::operator==(const _{{ util.class_name(cls) }}_& other) const {
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

bool Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_::operator<(const _{{ util.class_name(cls) }}_& other) const {
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

using _{{ util.class_name(cls) }}_ =  Const{{ util.class_name(cls) }}::_{{ util.class_name(cls) }}_;
Const{{ util.class_name(cls) }}::Const{{ util.class_name(cls) }}(const ::std::shared_ptr<::std::unique_ptr<_{{ util.class_name(cls) }}_>>& data): data_(data) {}
Const{{ util.class_name(cls) }}::Const{{ util.class_name(cls) }}(): data_(::std::make_shared<::std::unique_ptr<_{{ util.class_name(cls) }}_>>()) {}
Const{{ util.class_name(cls) }}::Const{{ util.class_name(cls) }}(const {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}& proto_{{ util.class_name(cls).lower() }}) {
  BuildFromProto(proto_{{ util.class_name(cls).lower() }});
}
Const{{ util.class_name(cls) }}::Const{{ util.class_name(cls) }}(const Const{{ util.class_name(cls) }}&) = default;
Const{{ util.class_name(cls) }}::Const{{ util.class_name(cls) }}(Const{{ util.class_name(cls) }}&&) noexcept = default;
Const{{ util.class_name(cls) }}::~Const{{ util.class_name(cls) }}() = default;

void Const{{ util.class_name(cls) }}::ToProto(PbMessage* proto_{{ util.class_name(cls).lower() }}) const {
  __SharedPtrOrDefault__()->ToProto(dynamic_cast<{{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}*>(proto_{{ util.class_name(cls).lower() }}));
}
  
::std::string Const{{ util.class_name(cls) }}::DebugString() const {
  return __SharedPtrOrDefault__()->DebugString();
}

bool Const{{ util.class_name(cls) }}::__Empty__() const {
  return !*data_;
}

int Const{{ util.class_name(cls) }}::FieldNumber4FieldName(const ::std::string& field_name) const  {
{% if util.has_message_type_fields(cls) %}
  static const ::std::map<::std::string, int> field_name2field_number{
{% for field in util.message_type_fields(cls) %}
    {"{{ util.field_name(field) }}", {{ util.field_number(field) }}},
{% endfor %}{# field #}
  };
  const auto& iter = field_name2field_number.find(field_name);
  if (iter != field_name2field_number.end()) { return iter->second; }
  return 0;
{% else %}
  return 0;
{% endif %}{# has message type fields #}
}

bool Const{{ util.class_name(cls) }}::FieldDefined4FieldNumber(int field_number) const  {
{% if util.has_message_type_fields(cls) %}
  switch (field_number) {
{% for field in util.message_type_fields(cls) %}
    case {{ util.field_number(field) }}:
{% endfor %}{# field #}
      return true;
    default:
      return false;
  }
{% else %}
  return false;
{% endif %}{# has message type fields #}
}

const ::std::set<::std::type_index>& Const{{ util.class_name(cls) }}::ValidTypeIndices4FieldNumber(int field_number) const {
  switch (field_number) {
{% for field in util.message_type_fields(cls) %}
    case {{ util.field_number(field) }}: {
      static const ::std::set<::std::type_index> type_indices{
{% if util.field_has_repeated_label(field) %}
        typeid(::oneflow::cfg::_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>)
{% elif util.field_has_map_label(field) %}
        typeid(::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>)
{% else %}
        typeid({{ util.field_type_name_with_cfg_namespace(field) }}),
{% if util.field_is_message_type(field) %}
        typeid(::oneflow::cfg::Message),
        typeid({{ util.field_type_name_const_with_cfg_namespace(field) }}),
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

const void* Const{{ util.class_name(cls) }}::FieldPtr4FieldNumber(int field_number) const  {
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
bool Const{{ util.class_name(cls) }}::has_{{ util.field_name(field) }}() const {
  return __SharedPtrOrDefault__()->has_{{ util.field_name(field) }}();
}
const {{ util.field_type_name_with_cfg_namespace(field) }}& Const{{ util.class_name(cls) }}::{{ util.field_name(field) }}() const {
  return __SharedPtrOrDefault__()->{{ util.field_name(field) }}();
}
// used by pybind11 only
{% if util.field_is_message_type(field) %}
::std::shared_ptr<{{ util.field_type_name_const_with_cfg_namespace(field) }}> Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }}() const {
  return {{ util.field_name(field) }}().__SharedConst__();
}
{% endif %}
{% elif util.field_has_repeated_label(field) %}
// repeated field {{ util.field_name(field) }}
::std::size_t Const{{ util.class_name(cls) }}::{{ util.field_name(field) }}_size() const {
  return __SharedPtrOrDefault__()->{{ util.field_name(field) }}_size();
}
const {{ util.field_repeated_container_name(field) }}& Const{{ util.class_name(cls) }}::{{ util.field_name(field) }}() const {
  return __SharedPtrOrDefault__()->{{ util.field_name(field) }}();
}
const {{ util.field_type_name_with_cfg_namespace(field) }}& Const{{ util.class_name(cls) }}::{{ util.field_name(field) }}(::std::size_t index) const {
  return __SharedPtrOrDefault__()->{{ util.field_name(field) }}(index);
}
// used by pybind11 only
::std::shared_ptr<Const{{ util.field_repeated_container_name(field) }}> Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }}() const {
  return {{ util.field_name(field) }}().__SharedConst__();
}
{% if util.field_is_message_type(field) %}
::std::shared_ptr<{{ util.field_type_name_const_with_cfg_namespace(field) }}> Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }}(::std::size_t index) const {
  return {{ util.field_name(field) }}(index).__SharedConst__();
}
{% else %}
{% endif %}{# field message type #}
{% elif util.field_has_oneof_label(field) %}
 // oneof field {{ util.oneof_name_of_oneof_type_field(field) }}: {{ util.field_name(field) }}
bool Const{{ util.class_name(cls) }}::has_{{ util.field_name(field) }}() const {
  return __SharedPtrOrDefault__()->has_{{ util.field_name(field) }}();
}
const {{ util.field_type_name_with_cfg_namespace(field) }}& Const{{ util.class_name(cls) }}::{{ util.field_name(field) }}() const {
  return __SharedPtrOrDefault__()->{{ util.field_name(field) }}();
}

// used by pybind11 only
{% if util.field_is_message_type(field) %}
::std::shared_ptr<{{ util.field_type_name_const_with_cfg_namespace(field) }}> Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }}() const {
  return {{ util.field_name(field) }}().__SharedConst__();
}
{% endif %}{# field message type #}
{# map begin#}
{% elif util.field_has_map_label(field) %}
// map field {{ util.field_name(field) }}
::std::size_t Const{{ util.class_name(cls) }}::{{ util.field_name(field) }}_size() const {
  return __SharedPtrOrDefault__()->{{ util.field_name(field) }}_size();
}

const {{ util.field_map_container_name(field) }}& Const{{ util.class_name(cls) }}::{{ util.field_name(field) }}() const {
  return __SharedPtrOrDefault__()->{{ util.field_name(field) }}();
}

// used by pybind11 only
::std::shared_ptr<Const{{ util.field_map_container_name(field) }}> Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }}() const {
  return {{ util.field_name(field) }}().__SharedConst__();
}
{# map end#}
{% endif %}{# field label type #}
{% endfor %}{# field #}
{% for oneof in util.message_type_oneofs(cls) %}
Const{{ util.class_name(cls) }}::{{ util.oneof_enum_name(oneof) }} Const{{ util.class_name(cls) }}::{{ util.oneof_name(oneof) }}_case() const {
  return __SharedPtrOrDefault__()->{{ util.oneof_name(oneof) }}_case();
}

bool Const{{ util.class_name(cls) }}::has_{{ util.oneof_name(oneof) }}() const {
  return __SharedPtrOrDefault__()->has_{{ util.oneof_name(oneof) }}();
}
{% endfor %}{# oneofs #}

::std::shared_ptr<Const{{ util.class_name(cls) }}> Const{{ util.class_name(cls) }}::__SharedConst__() const {
  return ::std::make_shared<Const{{ util.class_name(cls) }}>(data_);
}
int64_t Const{{ util.class_name(cls) }}::__Id__() const { return reinterpret_cast<int64_t>(data_.get()); }


bool Const{{ util.class_name(cls) }}::operator==(const Const{{ util.class_name(cls) }}& other) const {
  return *__SharedPtrOrDefault__() == *other.__SharedPtrOrDefault__();
}

bool Const{{ util.class_name(cls) }}::operator<(const Const{{ util.class_name(cls) }}& other) const {
  return *__SharedPtrOrDefault__() < *other.__SharedPtrOrDefault__();
}

const ::std::unique_ptr<_{{ util.class_name(cls) }}_>& Const{{ util.class_name(cls) }}::__SharedPtrOrDefault__() const {
  if (*data_) { return *data_; }
  static const ::std::unique_ptr<_{{ util.class_name(cls) }}_> default_ptr(new _{{ util.class_name(cls) }}_());
  return default_ptr;
}
const ::std::unique_ptr<_{{ util.class_name(cls) }}_>& Const{{ util.class_name(cls) }}::__SharedPtr__() {
  return *__SharedUniquePtr__();
}
const ::std::shared_ptr<::std::unique_ptr<_{{ util.class_name(cls) }}_>>& Const{{ util.class_name(cls) }}::__SharedUniquePtr__() {
  if (!*data_) { data_->reset(new _{{ util.class_name(cls) }}_()); }
  return data_;
}
// use a protected member method to avoid someone change member variable(data_) by Const{{ util.class_name(cls) }}
void Const{{ util.class_name(cls) }}::BuildFromProto(const PbMessage& proto_{{ util.class_name(cls).lower() }}) {
  data_ = ::std::make_shared<::std::unique_ptr<_{{ util.class_name(cls) }}_>>(new _{{ util.class_name(cls) }}_(dynamic_cast<const {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}&>(proto_{{ util.class_name(cls).lower() }})));
}

{{ util.class_name(cls) }}::{{ util.class_name(cls) }}(const ::std::shared_ptr<::std::unique_ptr<_{{ util.class_name(cls) }}_>>& data)
  : Const{{ util.class_name(cls) }}(data) {}
{{ util.class_name(cls) }}::{{ util.class_name(cls) }}(const {{ util.class_name(cls) }}& other) { CopyFrom(other); }
// enable nothrow for ::std::vector<{{ util.class_name(cls) }}> resize
{{ util.class_name(cls) }}::{{ util.class_name(cls) }}({{ util.class_name(cls) }}&&) noexcept = default; 
{{ util.class_name(cls) }}::{{ util.class_name(cls) }}(const {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}& proto_{{ util.class_name(cls).lower() }}) {
  InitFromProto(proto_{{ util.class_name(cls).lower() }});
}
{{ util.class_name(cls) }}::{{ util.class_name(cls) }}() = default;

{{ util.class_name(cls) }}::~{{ util.class_name(cls) }}() = default;

void {{ util.class_name(cls) }}::InitFromProto(const PbMessage& proto_{{ util.class_name(cls).lower() }}) {
  BuildFromProto(proto_{{ util.class_name(cls).lower() }});
}
  
void* {{ util.class_name(cls) }}::MutableFieldPtr4FieldNumber(int field_number) {
  switch (field_number) {
{% for field in util.message_type_fields(cls) %}
    case {{ util.field_number(field) }}: return mutable_{{ util.field_name(field) }}();
{% endfor %}{# field #}
    default: return nullptr;
  }
}

bool {{ util.class_name(cls) }}::operator==(const {{ util.class_name(cls) }}& other) const {
  return *__SharedPtrOrDefault__() == *other.__SharedPtrOrDefault__();
}

bool {{ util.class_name(cls) }}::operator<(const {{ util.class_name(cls) }}& other) const {
  return *__SharedPtrOrDefault__() < *other.__SharedPtrOrDefault__();
}
void {{ util.class_name(cls) }}::Clear() {
  if (data_) { data_->reset(); }
}
void {{ util.class_name(cls) }}::CopyFrom(const {{ util.class_name(cls) }}& other) {
  if (other.__Empty__()) {
    Clear();
  } else {
    __SharedPtr__()->CopyFrom(**other.data_);
  }
}
{{ util.class_name(cls) }}& {{ util.class_name(cls) }}::operator=(const {{ util.class_name(cls) }}& other) {
  CopyFrom(other);
  return *this;
}

{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
// required or optional field {{ util.field_name(field) }}
void {{ util.class_name(cls) }}::clear_{{ util.field_name(field) }}() {
  return __SharedPtr__()->clear_{{ util.field_name(field) }}();
}
{% if util.field_is_message_type(field) %}
{{ util.field_type_name_with_cfg_namespace(field) }}* {{ util.class_name(cls) }}::mutable_{{ util.field_name(field) }}() {
  return __SharedPtr__()->mutable_{{ util.field_name(field) }}();
}
// used by pybind11 only
::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> {{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }}() {
  return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
}
{% else %}
void {{ util.class_name(cls) }}::set_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value) {
  return __SharedPtr__()->set_{{ util.field_name(field) }}(value);
}
{{ util.field_type_name_with_cfg_namespace(field) }}* {{ util.class_name(cls) }}::mutable_{{ util.field_name(field) }}() {
  return  __SharedPtr__()->mutable_{{ util.field_name(field) }}();
}
{% endif %}
{% elif util.field_has_repeated_label(field) %}
// repeated field {{ util.field_name(field) }}
void {{ util.class_name(cls) }}::clear_{{ util.field_name(field) }}() {
  return __SharedPtr__()->clear_{{ util.field_name(field) }}();
}
{{ util.field_repeated_container_name(field) }}* {{ util.class_name(cls) }}::mutable_{{ util.field_name(field) }}() {
  return __SharedPtr__()->mutable_{{ util.field_name(field) }}();
}
{{ util.field_type_name_with_cfg_namespace(field) }}* {{ util.class_name(cls) }}::mutable_{{ util.field_name(field) }}(::std::size_t index) {
  return __SharedPtr__()->mutable_{{ util.field_name(field) }}(index);
}
{% if util.field_is_message_type(field) %}
// used by pybind11 only
::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> {{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }}() {
  return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
}
::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> {{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }}(::std::size_t index) {
  return mutable_{{ util.field_name(field) }}(index)->__SharedMutable__();
}
{{ util.field_type_name_with_cfg_namespace(field) }}* {{ util.class_name(cls) }}::add_{{ util.field_name(field) }}() {
  return __SharedPtr__()->add_{{ util.field_name(field) }}();
}
{% else %}
void {{ util.class_name(cls) }}::add_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value) {
  return __SharedPtr__()->add_{{ util.field_name(field) }}(value);
}
// used by pybind11 only
::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> {{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }}() {
  return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
}
{% endif %}{# field message type #}
{% elif util.field_has_oneof_label(field) %}
void {{ util.class_name(cls) }}::clear_{{ util.field_name(field) }}() {
  return __SharedPtr__()->clear_{{ util.field_name(field) }}();
}
{% if util.field_is_message_type(field) %}
{{ util.field_type_name_with_cfg_namespace(field) }}* {{ util.class_name(cls) }}::mutable_{{ util.field_name(field) }}() {
  return __SharedPtr__()->mutable_{{ util.field_name(field) }}();
}
// used by pybind11 only
::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> {{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }}() {
  return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
}
{% else %}
void {{ util.class_name(cls) }}::set_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value) {
  return __SharedPtr__()->set_{{ util.field_name(field) }}(value);
}
{{ util.field_type_name_with_cfg_namespace(field) }}* {{ util.class_name(cls) }}::mutable_{{ util.field_name(field) }}() {
  return  __SharedPtr__()->mutable_{{ util.field_name(field) }}();
}
{% endif %}{# field message type #}
{# map begin#}
{% elif util.field_has_map_label(field) %}
// repeated field {{ util.field_name(field) }}
void {{ util.class_name(cls) }}::clear_{{ util.field_name(field) }}() {
  return __SharedPtr__()->clear_{{ util.field_name(field) }}();
}

const {{ util.field_map_container_name(field) }} & {{ util.class_name(cls) }}::{{ util.field_name(field) }}() {
  return __SharedPtr__()->{{ util.field_name(field) }}();
}

{{ util.field_map_container_name(field) }}* {{ util.class_name(cls) }}::mutable_{{ util.field_name(field) }}() {
  return __SharedPtr__()->mutable_{{ util.field_name(field) }}();
}

  // used by pybind11 only
::std::shared_ptr<{{ util.field_map_container_name(field) }}> {{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }}() {
  return mutable_{{ util.field_name(field) }}()->__SharedMutable__();
}
{# map end#}
{% endif %}{# field label type #}
{% endfor %}{# field #}

::std::shared_ptr<{{ util.class_name(cls) }}> {{ util.class_name(cls) }}::__SharedMutable__() {
  return ::std::make_shared<{{ util.class_name(cls) }}>(__SharedUniquePtr__());
}
{% endif %}{# cls is not entry #}
{% endfor %}{# cls #}

{% for cls in util.module_nested_message_types(module) %}
{% if not util.class_is_map_entry(cls) %}
{% for field in util.message_type_fields(cls) %}
{# no duplicated class defined for each repeated field type #}
{% if util.field_has_repeated_label(field) and util.add_declared_repeated_field_type_name(field) %}
Const{{ util.field_repeated_container_name(field) }}::Const{{ util.field_repeated_container_name(field) }}(const ::std::shared_ptr<::std::vector<{{ util.field_type_name_with_cfg_namespace(field) }}>>& data): ::oneflow::cfg::_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>(data) {}
Const{{ util.field_repeated_container_name(field) }}::Const{{ util.field_repeated_container_name(field) }}() = default;
Const{{ util.field_repeated_container_name(field) }}::~Const{{ util.field_repeated_container_name(field) }}() = default;


bool Const{{ util.field_repeated_container_name(field) }}::operator==(const Const{{ util.field_repeated_container_name(field) }}& other) const {
  return *__SharedPtr__() == *other.__SharedPtr__();
}
bool Const{{ util.field_repeated_container_name(field) }}::operator<(const Const{{ util.field_repeated_container_name(field) }}& other) const {
  return *__SharedPtr__() < *other.__SharedPtr__();
}
::std::shared_ptr<Const{{ util.field_repeated_container_name(field) }}> Const{{ util.field_repeated_container_name(field) }}::__SharedConst__() const {
  return ::std::make_shared<Const{{ util.field_repeated_container_name(field) }}>(__SharedPtr__());
}
{% if util.field_is_message_type(field) %}
  ::std::shared_ptr<{{ util.field_type_name_const_with_cfg_namespace(field) }}> Const{{ util.field_repeated_container_name(field) }}::__SharedConst__(::std::size_t index) const {
    return Get(index).__SharedConst__();
  }
{% endif %}{# message_type #}

{{ util.field_repeated_container_name(field) }}::{{ util.field_repeated_container_name(field) }}(const ::std::shared_ptr<::std::vector<{{ util.field_type_name_with_cfg_namespace(field) }}>>& data): Const{{ util.field_repeated_container_name(field) }}(data) {}
{{ util.field_repeated_container_name(field) }}::{{ util.field_repeated_container_name(field) }}() = default;
{{ util.field_repeated_container_name(field) }}::~{{ util.field_repeated_container_name(field) }}() = default;

void {{ util.field_repeated_container_name(field) }}::CopyFrom(const Const{{ util.field_repeated_container_name(field) }}& other) {
  ::oneflow::cfg::_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::CopyFrom(other);
}
void {{ util.field_repeated_container_name(field) }}::CopyFrom(const {{ util.field_repeated_container_name(field) }}& other) {
  ::oneflow::cfg::_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::CopyFrom(other);
}
bool {{ util.field_repeated_container_name(field) }}::operator==(const {{ util.field_repeated_container_name(field) }}& other) const {
  return *__SharedPtr__() == *other.__SharedPtr__();
}
bool {{ util.field_repeated_container_name(field) }}::operator<(const {{ util.field_repeated_container_name(field) }}& other) const {
  return *__SharedPtr__() < *other.__SharedPtr__();
}
// used by pybind11 only
::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> {{ util.field_repeated_container_name(field) }}::__SharedMutable__() {
  return ::std::make_shared<{{ util.field_repeated_container_name(field) }}>(__SharedPtr__());
}
{% if util.field_is_message_type(field) %}
::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> {{ util.field_repeated_container_name(field) }}::__SharedAdd__() {
  return Add()->__SharedMutable__();
}
::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> {{ util.field_repeated_container_name(field) }}::__SharedMutable__(::std::size_t index) {
  return Mutable(index)->__SharedMutable__();
}
{% endif %}{# message_type #}
{% endif  %}{# repeated #}
{# map begin #}
{% if util.field_has_map_label(field) and util.add_declared_map_field_type_name(field) %}
Const{{ util.field_map_container_name(field) }}::Const{{ util.field_map_container_name(field) }}(const ::std::shared_ptr<::std::map<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>>& data): ::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>(data) {}
Const{{ util.field_map_container_name(field) }}::Const{{ util.field_map_container_name(field) }}() = default;
Const{{ util.field_map_container_name(field) }}::~Const{{ util.field_map_container_name(field) }}() = default;

bool Const{{ util.field_map_container_name(field) }}::operator==(const Const{{ util.field_map_container_name(field) }}& other) const {
  return *__SharedPtr__() == *other.__SharedPtr__();
}
bool Const{{ util.field_map_container_name(field) }}::operator<(const Const{{ util.field_map_container_name(field) }}& other) const {
  return *__SharedPtr__() < *other.__SharedPtr__();
}
// used by pybind11 only
const {{ util.field_map_value_type_name_with_cfg_namespace(field) }}& Const{{ util.field_map_container_name(field) }}::Get(const {{ util.field_map_key_type_name(field) }}& key) const {
return at(key);
}

// used by pybind11 only
::std::shared_ptr<Const{{ util.field_map_container_name(field) }}> Const{{ util.field_map_container_name(field) }}::__SharedConst__() const {
  return ::std::make_shared<Const{{ util.field_map_container_name(field) }}>(__SharedPtr__());
}

{% if util.field_is_message_type(util.field_map_value_type(field)) %}
// used by pybind11 only
::std::shared_ptr<Const{{ util.field_map_value_type_name(field) }}> Const{{ util.field_map_container_name(field) }}::__SharedConst__(const {{ util.field_map_key_type_name(field) }}& key) const {
  return at(key).__SharedConst__();
}

// ensuring mapped data's lifetime safety
::oneflow::cfg::_SharedConstPairIterator_<Const{{ util.field_map_container_name(field) }}, Const{{ util.field_map_value_type_name(field) }}> Const{{ util.field_map_container_name(field) }}::shared_const_begin() { return begin(); }
// ensuring mapped data's lifetime safety
::oneflow::cfg::_SharedConstPairIterator_<Const{{ util.field_map_container_name(field) }}, Const{{ util.field_map_value_type_name(field) }}> Const{{ util.field_map_container_name(field) }}::shared_const_end() { return end(); }
{% endif %}{# message_type #}

{{ util.field_map_container_name(field) }}::{{ util.field_map_container_name(field) }}(const ::std::shared_ptr<::std::map<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>>& data): Const{{ util.field_map_container_name(field) }}(data) {}
{{ util.field_map_container_name(field) }}::{{ util.field_map_container_name(field) }}() = default;
{{ util.field_map_container_name(field) }}::~{{ util.field_map_container_name(field) }}() = default;

void {{ util.field_map_container_name(field) }}::CopyFrom(const Const{{ util.field_map_container_name(field) }}& other) {
  ::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::CopyFrom(other);
}
void {{ util.field_map_container_name(field) }}::CopyFrom(const {{ util.field_map_container_name(field) }}& other) {
  ::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::CopyFrom(other);
}
bool {{ util.field_map_container_name(field) }}::operator==(const {{ util.field_map_container_name(field) }}& other) const {
  return *__SharedPtr__() == *other.__SharedPtr__();
}
bool {{ util.field_map_container_name(field) }}::operator<(const {{ util.field_map_container_name(field) }}& other) const {
  return *__SharedPtr__() < *other.__SharedPtr__();
}
// used by pybind11 only
::std::shared_ptr<{{ util.field_map_container_name(field) }}> {{ util.field_map_container_name(field) }}::__SharedMutable__() {
  return ::std::make_shared<{{ util.field_map_container_name(field) }}>(__SharedPtr__());
}

{% if util.field_is_message_type(util.field_map_value_type(field)) %}
::std::shared_ptr<{{ util.field_map_value_type_name_with_cfg_namespace(field) }}> {{ util.field_map_container_name(field) }}::__SharedMutable__(const {{ util.field_map_key_type_name(field) }}& key) {
  return (*this)[key].__SharedMutable__();
}
// ensuring mapped data's lifetime safety
::oneflow::cfg::_SharedMutPairIterator_<{{ util.field_map_container_name(field) }}, {{ util.field_map_value_type_name_with_cfg_namespace(field) }}> {{ util.field_map_container_name(field) }}::shared_mut_begin() { return begin(); }
// ensuring mapped data's lifetime safety
::oneflow::cfg::_SharedMutPairIterator_<{{ util.field_map_container_name(field) }}, {{ util.field_map_value_type_name_with_cfg_namespace(field) }}> {{ util.field_map_container_name(field) }}::shared_mut_end() { return end(); }
{% else %}
void {{ util.field_map_container_name(field) }}::Set(const {{ util.field_map_key_type_name(field) }}& key, const {{ util.field_map_value_type_name_with_cfg_namespace(field) }}& value) {
  (*this)[key] = value;
}
{% endif %}{# message_type #}
{# message_type #}
{% endif  %}{# map end #}
{% endfor %}{# field #}
{% endif %}{# cls is not entry #}
{% endfor %}{# cls #}

}
{% for package in util.module_package_list(module) %}
} // namespace {{ package }}
{% endfor %}{# package #}
