#ifndef {{ util.module_header_macro_lock(module) }}
#define {{ util.module_header_macro_lock(module) }}

#include <memory>
#include <vector>
#include <map>
#include <google/protobuf/message.h>
#include "oneflow/cfg/repeated_field.h"
#include "oneflow/cfg/map_field.h"
#include "oneflow/cfg/message.h"
#include "oneflow/cfg/shared_pair_iterator.h"

// forward declare enum defined in other module
{% for namespaces, cls in  util.other_file_declared_namespaces_and_enum_name(module) %}
{% for ns in namespaces %}
namespace {{ ns }} {
{% endfor %}
namespace cfg {
enum {{ cls }} : unsigned int;
}
{% for namespace in namespaces %}
}
{% endfor %}
{% endfor %}

// forward declare class defined in other module
{% for namespaces, cls in  util.other_file_declared_namespaces_and_class_name(module) %}
{% for ns in namespaces %}
namespace {{ ns }} {
{% endfor %}
namespace cfg {
class Const{{ cls }};
class {{ cls }};
}
{% for namespace in namespaces %}
}
{% endfor %}
{% endfor %}

{% for package in util.module_package_list(module) %}
namespace {{ package }} {
{% endfor %}

// forward declare proto class;
{% for cls in util.module_nested_message_types(module) %}
class {{ util.class_name(cls) }};
{% endfor %}

namespace cfg {

{% for cls in util.module_nested_message_types(module) %}
{% if not util.class_is_map_entry(cls) %}

class {{ util.class_name(cls) }};
class Const{{ util.class_name(cls) }};
{% for enm in util.message_type_enums(cls) %}
enum {{ util.enum_name(enm) }} : unsigned int {
{% for value in util.enum_values(enm) %}
  {{ util.enum_value_name(value) }} = {{ util.enum_value_number(value) }},
{% endfor %}
};

{% endfor %}{# oneof enum #}
{% endif %}{# cls is not entry #}
{% endfor %}{# cls #}

{% for enm in util.module_enum_types(module) %}
enum {{ util.enum_name(enm) }} : unsigned int {
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

{% for cls in util.module_nested_message_types(module) %}
{% if not util.class_is_map_entry(cls) %}
{% for field in util.message_type_fields(cls) %}
{# no duplicated class declared for each repeated field type #}
{% if util.field_has_repeated_label(field) and util.add_declared_repeated_field_type_name(field) %}
class {{ util.field_repeated_container_name(field) }};
class Const{{ util.field_repeated_container_name(field) }};
{% endif  %}{# repeated #}
{# map begin #}
{% if util.field_has_map_label(field) and util.add_declared_map_field_type_name(field) %}
class {{ util.field_map_container_name(field) }}; 
class Const{{ util.field_map_container_name(field) }};
{% endif  %}{# map end #}
{% endfor %}{# field #}

class Const{{ util.class_name(cls) }} : public ::oneflow::cfg::Message {
 public:
{% for oneof in util.message_type_oneofs(cls) %}

 // oneof enum {{ util.oneof_name(oneof) }}
 enum {{ util.oneof_enum_name(oneof) }} : unsigned int {
  {{ util.oneof_name(oneof).upper() }}_NOT_SET = 0,
  {% for field in util.oneof_type_fields(oneof) %}
  {{ util.oneof_type_field_enum_value_name(field) }} = {{ util.field_number(field) }},
  {% endfor %}
 };
{% endfor %}{# oneof enum #}

  class _{{ util.class_name(cls) }}_ {
   public:
    _{{ util.class_name(cls) }}_();
    explicit _{{ util.class_name(cls) }}_(const _{{ util.class_name(cls) }}_& other);
    explicit _{{ util.class_name(cls) }}_(_{{ util.class_name(cls) }}_&& other);
    _{{ util.class_name(cls) }}_(const {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}& proto_{{ util.class_name(cls).lower() }});
    ~_{{ util.class_name(cls) }}_();

    void InitFromProto(const {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}& proto_{{ util.class_name(cls).lower() }});

    void ToProto({{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}* proto_{{ util.class_name(cls).lower() }}) const;
    ::std::string DebugString() const;

    void Clear();
    void CopyFrom(const _{{ util.class_name(cls) }}_& other);
  {% for field in util.message_type_fields(cls) %}

  {% if util.field_has_required_or_optional_label(field) %}
    // optional field {{ util.field_name(field) }}
  {% if util.field_is_message_type(field) %}
   public:
    bool has_{{ util.field_name(field) }}() const;
    const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}() const;
    void clear_{{ util.field_name(field) }}();
    {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}();
   protected:
    bool has_{{ util.field_name(field) }}_ = false;
    ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> {{ util.field_name(field) }}_;
  {% else %}
   public:
    bool has_{{ util.field_name(field) }}() const;
    const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}() const;
    void clear_{{ util.field_name(field) }}();
    void set_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value);
    {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}();
   protected:
    bool has_{{ util.field_name(field) }}_ = false;
    {{ util.field_type_name_with_cfg_namespace(field) }} {{ util.field_name(field) }}_;
  {% endif %}{# field_type #}
  {% elif util.field_has_repeated_label(field) %}
    // repeated field {{ util.field_name(field) }}
   public:
    ::std::size_t {{ util.field_name(field) }}_size() const;
    const {{ util.field_repeated_container_name(field) }}& {{ util.field_name(field) }}() const;
    const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}(::std::size_t index) const;
    void clear_{{ util.field_name(field) }}();
    {{ util.field_repeated_container_name(field) }}* mutable_{{ util.field_name(field) }}();
    {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}(::std::size_t index);
  {% if util.field_is_message_type(field) %}
    {{ util.field_type_name_with_cfg_namespace(field) }}* add_{{ util.field_name(field) }}();
  {% else %}
    void add_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value);
  {% endif %}{# field message type #}
   protected:
    ::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> {{ util.field_name(field) }}_;
  {% elif util.field_has_oneof_label(field) %}
   // oneof field {{ util.oneof_name_of_oneof_type_field(field) }}: {{ util.field_name(field) }}
   public:
    bool has_{{ util.field_name(field) }}() const;
    void clear_{{ util.field_name(field) }}();
    const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}() const;
  {% if util.field_is_message_type(field) %}
    {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}();
  {% else %}
    void set_{{ util.field_name(field) }}(const {{util.field_type_name_with_cfg_namespace(field) }}& value);
    {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}();
  {% endif %}{# field message type #}
  {% elif util.field_has_map_label(field) %}
   public:
    ::std::size_t {{ util.field_name(field) }}_size() const;
    const {{ util.field_map_container_name(field) }}& {{ util.field_name(field) }}() const;

    {{ util.field_map_container_name(field) }} * mutable_{{ util.field_name(field) }}();

    const {{ util.field_map_value_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}({{ util.field_map_key_type_name(field) }} key) const;

    void clear_{{ util.field_name(field) }}();
  {% if util.field_is_message_type(field) %}
  {% else %}
    void add_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value);
  {% endif %}{# field message type #}
   protected:
    ::std::shared_ptr<{{ util.field_map_container_name(field) }}> {{ util.field_name(field) }}_;
  {% endif %}{# label #}
  {% endfor %}{# field #}
  {% for oneof in util.message_type_oneofs(cls) %}
   
   public:
    // oneof {{ util.oneof_name(oneof) }}
    {{ util.oneof_enum_name(oneof) }} {{ util.oneof_name(oneof) }}_case() const;
    bool has_{{util.oneof_name(oneof)}}() const;
   protected:
    void clear_{{util.oneof_name(oneof)}}();
    void {{util.oneof_name(oneof)}}_copy_from(const _{{ util.class_name(cls) }}_& other);
    union {{ util.oneof_camel_name(oneof) }}Union {
      // 64-bit aligned
      uint64_t __{{util.oneof_name(oneof)}}_for_padding_64bit__;
  {% for field in util.oneof_type_fields(oneof) %}
  {% if util.field_is_message_type(field) %}
      char {{ util.field_name(field) }}_[sizeof(::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}>)];
  {% elif util.field_is_string_type(field) %}
      char {{ util.field_name(field) }}_[sizeof(::std::string)];
  {% else %}
      {{ util.field_scalar_type_name(field) }} {{ util.field_name(field) }}_;
  {% endif %}{# field message type #}
  {% endfor %}{# oneof_fields #}
    } {{ util.oneof_name(oneof) }}_;
    {{ util.oneof_enum_name(oneof) }} {{ util.oneof_name(oneof) }}_case_ = {{ util.oneof_name(oneof).upper() }}_NOT_SET;
  {% endfor %}{# message_oneof #}
   
   public:
    int compare(const _{{ util.class_name(cls) }}_& other);

    bool operator==(const _{{ util.class_name(cls) }}_& other) const;

    bool operator<(const _{{ util.class_name(cls) }}_& other) const;
  };

  Const{{ util.class_name(cls) }}(const ::std::shared_ptr<::std::unique_ptr<_{{ util.class_name(cls) }}_>>& data);
  Const{{ util.class_name(cls) }}(const Const{{ util.class_name(cls) }}&);
  Const{{ util.class_name(cls) }}(Const{{ util.class_name(cls) }}&&) noexcept;
  Const{{ util.class_name(cls) }}();
  Const{{ util.class_name(cls) }}(const {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}& proto_{{ util.class_name(cls).lower() }});
  ~Const{{ util.class_name(cls) }}() override;

  using PbMessage = ::google::protobuf::Message;
  void ToProto(PbMessage* proto_{{ util.class_name(cls).lower() }}) const override;
  
  ::std::string DebugString() const;

  bool __Empty__() const;

  int FieldNumber4FieldName(const ::std::string& field_name) const override;

  bool FieldDefined4FieldNumber(int field_number) const override;

  const ::std::set<::std::type_index>& ValidTypeIndices4FieldNumber(int field_number) const override;

  const void* FieldPtr4FieldNumber(int field_number) const override;

{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
  // required or optional field {{ util.field_name(field) }}
 public:
  bool has_{{ util.field_name(field) }}() const;
  const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}() const;
  // used by pybind11 only
{% if util.field_is_message_type(field) %}
  ::std::shared_ptr<{{ util.field_type_name_const_with_cfg_namespace(field) }}> shared_const_{{ util.field_name(field) }}() const;
{% endif %}
{% elif util.field_has_repeated_label(field) %}
  // repeated field {{ util.field_name(field) }}
 public:
  ::std::size_t {{ util.field_name(field) }}_size() const;
  const {{ util.field_repeated_container_name(field) }}& {{ util.field_name(field) }}() const;
  const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}(::std::size_t index) const;
  // used by pybind11 only
  ::std::shared_ptr<Const{{ util.field_repeated_container_name(field) }}> shared_const_{{ util.field_name(field) }}() const;
{% if util.field_is_message_type(field) %}
  ::std::shared_ptr<{{ util.field_type_name_const_with_cfg_namespace(field) }}> shared_const_{{ util.field_name(field) }}(::std::size_t index) const;
{% else %}
{% endif %}{# field message type #}
{% elif util.field_has_oneof_label(field) %}
 // oneof field {{ util.oneof_name_of_oneof_type_field(field) }}: {{ util.field_name(field) }}
 public:
  bool has_{{ util.field_name(field) }}() const;
  const {{ util.field_type_name_with_cfg_namespace(field) }}& {{ util.field_name(field) }}() const;
  // used by pybind11 only
{% if util.field_is_message_type(field) %}
  ::std::shared_ptr<{{ util.field_type_name_const_with_cfg_namespace(field) }}> shared_const_{{ util.field_name(field) }}() const;
{% endif %}{# field message type #}
{# map begin#}
{% elif util.field_has_map_label(field) %}
  // map field {{ util.field_name(field) }}
 public:
  ::std::size_t {{ util.field_name(field) }}_size() const;
  const {{ util.field_map_container_name(field) }}& {{ util.field_name(field) }}() const;

  // used by pybind11 only
  ::std::shared_ptr<Const{{ util.field_map_container_name(field) }}> shared_const_{{ util.field_name(field) }}() const;
{# map end#}
{% endif %}{# field label type #}
{% endfor %}{# field #}
{% for oneof in util.message_type_oneofs(cls) %}
  {{ util.oneof_enum_name(oneof) }} {{ util.oneof_name(oneof) }}_case() const;

  bool has_{{ util.oneof_name(oneof) }}() const;
{% endfor %}{# oneofs #}

  ::std::shared_ptr<Const{{ util.class_name(cls) }}> __SharedConst__() const;
  int64_t __Id__() const;
  // the data of `this` will be moved to the result which is mutable
  ::std::shared_ptr<{{ util.class_name(cls) }}> __Move__();
 public:
  bool operator==(const Const{{ util.class_name(cls) }}& other) const;

  bool operator<(const Const{{ util.class_name(cls) }}& other) const;
 protected:
  const ::std::unique_ptr<_{{ util.class_name(cls) }}_>& __SharedPtrOrDefault__() const;
  const ::std::unique_ptr<_{{ util.class_name(cls) }}_>& __SharedPtr__();
  const ::std::shared_ptr<::std::unique_ptr<_{{ util.class_name(cls) }}_>>& __SharedUniquePtr__();
  // use a protected member method to avoid someone change member variable(data_) by Const{{ util.class_name(cls) }}
  void BuildFromProto(const PbMessage& proto_{{ util.class_name(cls).lower() }});
  // use ::std::shared_ptr for sharing reference between mutable object and const object
  // use ::std::unique_ptr for moving ownership 
  ::std::shared_ptr<::std::unique_ptr<_{{ util.class_name(cls) }}_>> data_;
};

class {{ util.class_name(cls) }} final : public Const{{ util.class_name(cls) }} {
 public:
  {{ util.class_name(cls) }}(const ::std::shared_ptr<::std::unique_ptr<_{{ util.class_name(cls) }}_>>& data);
  {{ util.class_name(cls) }}(const {{ util.class_name(cls) }}& other);
  // enable nothrow for ::std::vector<{{ util.class_name(cls) }}> resize 
  {{ util.class_name(cls) }}({{ util.class_name(cls) }}&&) noexcept;
  {{ util.class_name(cls) }}();
  {{ util.class_name(cls) }}(const {{ util.module_package_namespace(module) }}::{{ util.class_name(cls) }}& proto_{{ util.class_name(cls).lower() }});

  ~{{ util.class_name(cls) }}();

  void InitFromProto(const PbMessage& proto_{{ util.class_name(cls).lower() }}) override;
  
  void* MutableFieldPtr4FieldNumber(int field_number) override;


  bool operator==(const {{ util.class_name(cls) }}& other) const;
  bool operator<(const {{ util.class_name(cls) }}& other) const;
  void Clear();
  void CopyFrom(const {{ util.class_name(cls) }}& other);
  {{ util.class_name(cls) }}& operator=(const {{ util.class_name(cls) }}& other);

{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
  // required or optional field {{ util.field_name(field) }}
 public:
  void clear_{{ util.field_name(field) }}();
{% if util.field_is_message_type(field) %}
  {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}();
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> shared_mutable_{{ util.field_name(field) }}();
{% else %}
  void set_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value);
  {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}();
{% endif %}
{% elif util.field_has_repeated_label(field) %}
  // repeated field {{ util.field_name(field) }}
 public:
  void clear_{{ util.field_name(field) }}();
  {{ util.field_repeated_container_name(field) }}* mutable_{{ util.field_name(field) }}();
  {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}(::std::size_t index);
{% if util.field_is_message_type(field) %}
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> shared_mutable_{{ util.field_name(field) }}();
  ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> shared_mutable_{{ util.field_name(field) }}(::std::size_t index);
  {{ util.field_type_name_with_cfg_namespace(field) }}* add_{{ util.field_name(field) }}();
{% else %}
  void add_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value);
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> shared_mutable_{{ util.field_name(field) }}();
{% endif %}{# field message type #}
{% elif util.field_has_oneof_label(field) %}
  void clear_{{ util.field_name(field) }}();
{% if util.field_is_message_type(field) %}
  {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}();
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> shared_mutable_{{ util.field_name(field) }}();
{% else %}
  void set_{{ util.field_name(field) }}(const {{ util.field_type_name_with_cfg_namespace(field) }}& value);
  {{ util.field_type_name_with_cfg_namespace(field) }}* mutable_{{ util.field_name(field) }}();
{% endif %}{# field message type #}
{# map begin#}
{% elif util.field_has_map_label(field) %}
  // repeated field {{ util.field_name(field) }}
 public:
  void clear_{{ util.field_name(field) }}();

  const {{ util.field_map_container_name(field) }} & {{ util.field_name(field) }}();

  {{ util.field_map_container_name(field) }}* mutable_{{ util.field_name(field) }}();

  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_map_container_name(field) }}> shared_mutable_{{ util.field_name(field) }}();
{# map end#}
{% endif %}{# field label type #}
{% endfor %}{# field #}

  ::std::shared_ptr<{{ util.class_name(cls) }}> __SharedMutable__();
};

{% endif %}{# cls is not entry #}
{% endfor %}{# cls #}

{% for cls in util.module_nested_message_types(module) %}
{% if not util.class_is_map_entry(cls) %}

{% for field in util.message_type_fields(cls) %}
{# no duplicated class defined for each repeated field type #}
{% if util.field_has_repeated_label(field) and util.add_defined_repeated_field_type_name(field) %}

// inheritance is helpful for avoiding container iterator boilerplate 
class Const{{ util.field_repeated_container_name(field) }} : public ::oneflow::cfg::_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}> {
 public:
  Const{{ util.field_repeated_container_name(field) }}(const ::std::shared_ptr<::std::vector<{{ util.field_type_name_with_cfg_namespace(field) }}>>& data);
  Const{{ util.field_repeated_container_name(field) }}();
  ~Const{{ util.field_repeated_container_name(field) }}();

  bool operator==(const Const{{ util.field_repeated_container_name(field) }}& other) const;
  bool operator<(const Const{{ util.field_repeated_container_name(field) }}& other) const;
  // used by pybind11 only
  ::std::shared_ptr<Const{{ util.field_repeated_container_name(field) }}> __SharedConst__() const;
{% if util.field_is_message_type(field) %}
  ::std::shared_ptr<{{ util.field_type_name_const_with_cfg_namespace(field) }}> __SharedConst__(::std::size_t index) const;
{% endif %}{# message_type #}
};
class {{ util.field_repeated_container_name(field) }} final : public Const{{ util.field_repeated_container_name(field) }} {
 public:
  {{ util.field_repeated_container_name(field) }}(const ::std::shared_ptr<::std::vector<{{ util.field_type_name_with_cfg_namespace(field) }}>>& data);
  {{ util.field_repeated_container_name(field) }}();
  ~{{ util.field_repeated_container_name(field) }}();
  void CopyFrom(const Const{{ util.field_repeated_container_name(field) }}& other);
  void CopyFrom(const {{ util.field_repeated_container_name(field) }}& other);
  bool operator==(const {{ util.field_repeated_container_name(field) }}& other) const;
  bool operator<(const {{ util.field_repeated_container_name(field) }}& other) const;
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> __SharedMutable__();
{% if util.field_is_message_type(field) %}
  ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> __SharedAdd__();
  ::std::shared_ptr<{{ util.field_type_name_with_cfg_namespace(field) }}> __SharedMutable__(::std::size_t index);
{% endif %}{# message_type #}
};
{% endif  %}{# repeated #}
{# map begin #}
{% if util.field_has_map_label(field) and util.add_defined_map_field_type_name(field) %}

// inheritance is helpful for avoid container iterator boilerplate 
class Const{{ util.field_map_container_name(field) }} : public ::oneflow::cfg::_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}> {
 public:
  Const{{ util.field_map_container_name(field) }}(const ::std::shared_ptr<::std::map<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>>& data);
  Const{{ util.field_map_container_name(field) }}();
  ~Const{{ util.field_map_container_name(field) }}();

  bool operator==(const Const{{ util.field_map_container_name(field) }}& other) const;
  bool operator<(const Const{{ util.field_map_container_name(field) }}& other) const;
  // used by pybind11 only
  const {{ util.field_map_value_type_name_with_cfg_namespace(field) }}& Get(const {{ util.field_map_key_type_name(field) }}& key) const;

  // used by pybind11 only
  ::std::shared_ptr<Const{{ util.field_map_container_name(field) }}> __SharedConst__() const;
{% if util.field_is_message_type(util.field_map_value_type(field)) %}
  // used by pybind11 only
  ::std::shared_ptr<Const{{ util.field_map_value_type_name(field) }}> __SharedConst__(const {{ util.field_map_key_type_name(field) }}& key) const;
  // used by pybind11 only
  using shared_const_iterator = ::oneflow::cfg::_SharedConstPairIterator_<Const{{ util.field_map_container_name(field) }}, Const{{ util.field_map_value_type_name(field) }}>;
  // ensuring mapped data's lifetime safety
  shared_const_iterator shared_const_begin();
  // ensuring mapped data's lifetime safety
  shared_const_iterator shared_const_end();
{% endif %}{# message_type #}
};
class {{ util.field_map_container_name(field) }} final : public Const{{ util.field_map_container_name(field) }} {
 public:
  {{ util.field_map_container_name(field) }}(const ::std::shared_ptr<::std::map<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>>& data);
  {{ util.field_map_container_name(field) }}();
  ~{{ util.field_map_container_name(field) }}();
  void CopyFrom(const Const{{ util.field_map_container_name(field) }}& other);
  void CopyFrom(const {{ util.field_map_container_name(field) }}& other);
  bool operator==(const {{ util.field_map_container_name(field) }}& other) const;
  bool operator<(const {{ util.field_map_container_name(field) }}& other) const;
  // used by pybind11 only
  ::std::shared_ptr<{{ util.field_map_container_name(field) }}> __SharedMutable__();

{% if util.field_is_message_type(util.field_map_value_type(field)) %}
  ::std::shared_ptr<{{ util.field_map_value_type_name_with_cfg_namespace(field) }}> __SharedMutable__(const {{ util.field_map_key_type_name(field) }}& key);
  // used by pybind11 only
  using shared_mut_iterator = ::oneflow::cfg::_SharedMutPairIterator_<{{ util.field_map_container_name(field) }}, {{ util.field_map_value_type_name_with_cfg_namespace(field) }}>;
  // ensuring mapped data's lifetime safety
  shared_mut_iterator shared_mut_begin();
  // ensuring mapped data's lifetime safety
  shared_mut_iterator shared_mut_end();
{% else %}
  void Set(const {{ util.field_map_key_type_name(field) }}& key, const {{ util.field_map_value_type_name_with_cfg_namespace(field) }}& value);
{% endif %}{# message_type #}
{# message_type #}
};
{% endif  %}{# map end #}
{% endfor %}{# field #}


inline ::std::shared_ptr<{{ util.class_name(cls) }}> Const{{ util.class_name(cls) }}::__Move__() {
  if (__Empty__()) { return ::std::make_shared<{{ util.class_name(cls) }}>(); }
  auto data = ::std::make_shared<::std::unique_ptr<_{{ util.class_name(cls) }}_>>();
  *data = ::std::move(*data_);
  return ::std::make_shared<{{ util.class_name(cls) }}>(data);
}
{% endif %}{# cls is not entry #}
{% endfor %}{# cls #}

} //namespace cfg

{% for package in util.module_package_list(module) %}
} // namespace {{ package }}
{% endfor %}{# package #}
#endif  // {{ util.module_header_macro_lock(module) }}
