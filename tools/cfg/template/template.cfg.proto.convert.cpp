#include "{{ util.module_cfg_convert_header_name(module) }}"

namespace oneflow {

{% for enm in util.module_enum_types(module) %}

::oneflow::{{ util.enum_name(enm) }} Cfg{{ util.enum_name(enm) }}ToProto{{ util.enum_name(enm) }}(const cfg::{{ util.enum_name(enm) }}& cfg_enum) {
  return ::oneflow::{{ util.enum_name(enm) }}(int(cfg_enum));
}

cfg::{{ util.enum_name(enm) }} Proto{{ util.enum_name(enm) }}ToCfg{{ util.enum_name(enm) }}(const ::oneflow::{{ util.enum_name(enm) }}& proto_enum) {
  return cfg::{{ util.enum_name(enm) }}(int(proto_enum));
}
{% endfor %}{# enms #}

{% for cls in util.module_message_types(module) %}

cfg::{{ util.class_name(cls) }} FromProto(const ::oneflow::{{ util.class_name(cls) }}& proto_{{ util.class_name(cls).lower() }}) {
  cfg::{{ util.class_name(cls) }} cfg_{{ util.class_name(cls).lower() }};
{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
  // required_or_optional field: {{ util.field_name(field) }}
  if (proto_{{ util.class_name(cls).lower() }}.has_{{ util.field_name(field) }}()) {
{% if util.field_is_message_type(field)%}
    cfg_{{ util.class_name(cls).lower() }}.mutable_{{ util.field_name(field) }}()->CopyFrom(FromProto(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()));
{% elif util.field_is_enum_type(field) %}
    cfg_{{ util.class_name(cls).lower() }}.set_{{ util.field_name(field) }}(Proto{{ util.field_enum_name(field) }}ToCfg{{ util.field_enum_name(field) }}(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()));
{% else %}
    cfg_{{ util.class_name(cls).lower() }}.set_{{ util.field_name(field) }}(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}());
{% endif %}{# message_type #}
  }
{% elif util.field_has_repeated_label(field) %}
  // repeated field: {{ util.field_name(field) }}
  if (!proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}.empty()) {
{% if util.field_is_message_type(field)%}
    for (const ::oneflow::{{ util.field_type_name(field) }}& elem : proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}() ) {
      *cfg_{{ util.class_name(cls).lower() }}.mutable_{{ util.field_name(field) }}()->Add() = FromProto(elem);
    }
{% elif util.field_is_enum_type(field) %}
    for (const int& elem : proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}() ) {
      cfg_{{ util.class_name(cls).lower() }}.add_{{ util.field_name(field) }}(cfg::{{ util.field_enum_name(field) }}(elem));
    }
{% else %}
    for (const {{ util.field_type_name(field) }}& elem : proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()) {
      cfg_{{ util.class_name(cls).lower() }}.add_{{ util.field_name(field) }}(elem);
    }
{% endif %}{# field_type #}
  }
{% elif util.field_is_map(field) %}
  // map field : {{ util.field_name(field) }}
  if (!proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}().empty()) {
    {{ util.field_map_container_name(field) }}& mut_{{ util.field_name(field) }} = *(proto_{{ util.class_name(cls).lower() }}.mutable_{{ util.field_name(field) }}());
    for (const auto& pair : proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()) {
{% if util.field_map_value_type_is_message(field) %}
      cfg_{{ util.class_name(cls).lower() }}.mut_{{ util.field_name(field) }}[pair.first] = FromProto(pair.second);
{% elif util.field_map_value_type_is_enum(field) %}
      cfg_{{ util.class_name(cls).lower() }}.mut_{{ util.field_name(field) }}[pair.first] = Proto{{ util.field_map_value_type_enum_name(field) }}ToCfg{{ util.field_map_value_type_enum_name(field) }}(pair.second);
{% else %}
      cfg_{{ util.class_name(cls).lower() }}.mut_{{ util.field_name(field) }}[pair.first] = pair.second;
{% endif %}{# map_value_type #}
    }
  }
{% endif %}{# field_label #}
{% endfor %}{# field #}
{% for oneof in util.message_type_oneofs(cls) %}
  // oneof field: {{ util.oneof_name(oneof) }}
  cfg::{{ util.class_name(cls) }}::{{ util.oneof_enum_name(oneof) }} {{ util.oneof_name(oneof) }}_case = cfg::{{ util.class_name(cls) }}::{{ util.oneof_enum_name(oneof) }}(int(proto_{{ util.class_name(cls).lower() }}.{{ util.oneof_name(oneof) }}_case()));
  switch ({{ util.oneof_name(oneof) }}_case) {
{% for field in util.oneof_type_fields(oneof) %}
    case cfg::{{ util.class_name(cls) }}::{{ util.oneof_type_field_enum_value_name(field) }}: {
{% if util.field_is_message_type(field) %}
      cfg_{{ util.class_name(cls).lower() }}.mutable_{{ util.field_name(field) }}()->CopyFrom(FromProto(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()));
{% elif util.field_is_enum_type(field) %}
      cfg_{{ util.class_name(cls).lower() }}.set_{{ util.field_name(field) }}(Proto{{ util.field_enum_name(field) }}ToCfg{{ util.field_enum_name(field) }}(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()));
{% else %}
      cfg_{{ util.class_name(cls).lower() }}.set_{{ util.field_name(field) }}(proto_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}());
{% endif %}{# message_type #}
      break;
    }
{% endfor %}{# oneof_field #}
    case cfg::{{ util.class_name(cls) }}::{{ util.oneof_name(oneof).upper() }}_NOT_SET: {
      break;
    }
  }
{% endfor %}{# oneofs #}
  return cfg_{{ util.class_name(cls).lower() }};
}

::oneflow::{{ util.class_name(cls) }} ToProto(const cfg::{{ util.class_name(cls) }}& cfg_{{ util.class_name(cls).lower() }}) {
  ::oneflow::{{ util.class_name(cls) }} proto_{{ util.class_name(cls).lower() }};
{% for field in util.message_type_fields(cls) %}
{% if util.field_has_required_or_optional_label(field) %}
// required_or_optional field: {{ util.field_name(field) }}
  if (cfg_{{ util.class_name(cls).lower() }}.has_{{ util.field_name(field) }}()) {
{% if util.field_is_message_type(field)%}
    proto_{{ util.class_name(cls).lower() }}.mutable_{{ util.field_name(field) }}()->CopyFrom(ToProto(cfg_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()));
{% elif util.field_is_enum_type(field) %}
    proto_{{ util.class_name(cls).lower() }}.set_{{ util.field_name(field) }}(Cfg{{ util.field_enum_name(field) }}ToProto{{ util.field_enum_name(field) }}(cfg_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()));
{% else %}
    proto_{{ util.class_name(cls).lower() }}.set_{{ util.field_name(field) }}(cfg_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}());
{% endif %}{# message_type #}
  }
{% elif util.field_has_repeated_label(field) %}
  // repeated field: {{ util.field_name(field) }}
  if (!cfg_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}.empty()) {
{% if util.field_is_message_type(field)%}
    for (const cfg::{{ util.field_type_name(field) }}& elem : cfg_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}() ) {
      *proto_{{ util.class_name(cls).lower() }}.mutable_{{ util.field_name(field) }}()->Add() = ToProto(elem);
    }
{% elif util.field_is_enum_type(field) %}
    for (const int& elem : cfg_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}() ) {
      proto_{{ util.class_name(cls).lower() }}.add_{{ util.field_name(field) }}(::oneflow::{{ util.field_enum_name(field) }}(elem));
    }
{% else %}
    for (const {{ util.field_type_name(field) }}& elem : cfg_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()) {
      proto_{{ util.class_name(cls).lower() }}.add_{{ util.field_name(field) }}(elem);
    }
{% endif %}{# field_type #}
  }
{% elif util.field_is_map(field) %}
  // map field : {{ util.field_name(field) }}
  if (!cfg_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}().empty()) {
    auto& mut_{{ util.field_name(field) }} = *(proto_{{ util.class_name(cls).lower() }}.mutable_{{ util.field_name(field) }}());
    for (const auto& pair : cfg_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()) {
{% if util.field_map_value_type_is_message(field) %}
      proto_{{ util.class_name(cls).lower() }}.mut_{{ util.field_name(field) }}[pair.first] = ToProto(pair.second);
{% elif util.field_map_value_type_is_enum(field) %}
      proto_{{ util.class_name(cls).lower() }}.mut_{{ util.field_name(field) }}[pair.first] = Cfg{{ util.field_map_value_type_enum_name(field) }}ToProto{{ util.field_map_value_type_enum_name(field) }}(pair.second);
{% else %}
      proto_{{ util.class_name(cls).lower() }}.mut_{{ util.field_name(field) }}[pair.first] = pair.second;
{% endif %}{# map_value_type #}
    }
  }
{% endif %}{# field_label #}
{% endfor %}{# field #}
{% for oneof in util.message_type_oneofs(cls) %}
  // oneof field: {{ util.oneof_name(oneof) }}
  ::oneflow::{{ util.class_name(cls) }}::{{ util.oneof_enum_name(oneof) }} {{ util.oneof_name(oneof) }}_case = ::oneflow::{{ util.class_name(cls) }}::{{ util.oneof_enum_name(oneof) }}(int(cfg_{{ util.class_name(cls).lower() }}.{{ util.oneof_name(oneof) }}_case()));
  switch ({{ util.oneof_name(oneof) }}_case) {
{% for field in util.oneof_type_fields(oneof) %}
    case ::oneflow::{{ util.class_name(cls) }}::{{ util.oneof_type_field_enum_value_name(field) }}: {
{% if util.field_is_message_type(field) %}
      proto_{{ util.class_name(cls).lower() }}.mutable_{{ util.field_name(field) }}()->CopyFrom(ToProto(cfg_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()));
{% elif util.field_is_enum_type(field) %}
      proto_{{ util.class_name(cls).lower() }}.set_{{ util.field_name(field) }}(Cfg{{ util.field_enum_name(field) }}ToProto{{ util.field_enum_name(field) }}(cfg_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}()));
{% else %}
      proto_{{ util.class_name(cls).lower() }}.set_{{ util.field_name(field) }}(cfg_{{ util.class_name(cls).lower() }}.{{ util.field_name(field) }}());
{% endif %}{# message_type #}
      break;
    }
{% endfor %}{# oneof_field #}
    case ::oneflow::{{ util.class_name(cls) }}::{{ util.oneof_name(oneof).upper() }}_NOT_SET: {
      break;
    }
  }
{% endfor %}{# oneofs #}
  return proto_{{ util.class_name(cls).lower() }};
}
{% endfor %}{# cls #}

} // namespace oneflow
