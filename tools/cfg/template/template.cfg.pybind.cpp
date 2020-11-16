#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include "oneflow/cfg/pybind_module_registry.h"
#include "{{ util.module_cfg_header_name(module) }}"
{% for dependency in util.module_dependencies(module) %}
#include "{{ util.module_cfg_header_name(dependency) }}"
{% endfor %}

ONEFLOW_CFG_PYBIND11_MODULE("{{ util.module_get_python_module_path(module) }}", m) {
{% if util.module_has_package(module) %}
  using namespace {{ "::".join(util.module_package_list(module)) }}::cfg;
{% else %}
  using namespace cfg;
{% endif %}
{% for enm in util.module_enum_types(module) %}
  {
    pybind11::enum_<{{ util.module_package_cfg_namespace(module) }}::{{ util.enum_name(enm) }}> enm(m, "{{ util.enum_name(enm) }}");
{% for value in util.enum_values(enm) %}
    enm.value("{{ util.enum_value_name(value) }}", {{ util.module_package_cfg_namespace(module) }}::{{ util.enum_value_name(value) }});
{% endfor %}{# enum_values #}
{% for value in util.enum_values(enm) %}
    m.attr("{{ util.enum_value_name(value) }}") = enm.attr("{{ util.enum_value_name(value) }}");
{% endfor %}{# enum_values #}
  }
{% endfor %}{# enum_types #}

{% for cls in util.module_nested_message_types(module) %}
{% if not util.class_is_map_entry(cls) %}
{% for enm in util.message_type_enums(cls) %}
  {
    pybind11::enum_<{{ util.module_package_cfg_namespace(module) }}::{{ util.enum_name(enm) }}> enm(m, "{{ util.enum_name(enm) }}");
{% for value in util.enum_values(enm) %}
    enm.value("{{ util.enum_value_name(value) }}", {{ util.module_package_cfg_namespace(module) }}::{{ util.enum_value_name(value) }});
{% endfor %}{# enum_values #}
{% for value in util.enum_values(enm) %}
    m.attr("{{ util.enum_value_name(value) }}") = enm.attr("{{ util.enum_value_name(value) }}");
{% endfor %}{# enum_values #}
  }
{% endfor %}{# enums #}
{% endif %}{# cls is not entry #}
{% endfor %}{# cls #}

{% for cls in util.module_nested_message_types(module) %}
{% if not util.class_is_map_entry(cls) %}
{% for field in util.message_type_fields(cls) %}
{# no duplicated python class registered for each repeated field type #}
{% if util.field_has_repeated_label(field) and util.add_declared_repeated_field_type_name(field) %}
  {
    pybind11::class_<Const{{ util.field_repeated_container_name(field) }}, std::shared_ptr<Const{{ util.field_repeated_container_name(field) }}>> registry(m, "Const{{ util.field_repeated_container_name(field) }}");
    registry.def("__len__", &Const{{ util.field_repeated_container_name(field) }}::size);
    registry.def(pybind11::self == pybind11:: self);
    registry.def(pybind11::self < pybind11:: self);
{% if util.field_is_message_type(field) %}
    registry.def("__getitem__", (::std::shared_ptr<Const{{ util.field_type_name(field) }}> (Const{{ util.field_repeated_container_name(field) }}::*)(::std::size_t) const)&Const{{ util.field_repeated_container_name(field) }}::__SharedConst__);
    registry.def("Get", (::std::shared_ptr<Const{{ util.field_type_name(field) }}> (Const{{ util.field_repeated_container_name(field) }}::*)(::std::size_t) const)&Const{{ util.field_repeated_container_name(field) }}::__SharedConst__);
{% else %}
    registry.def("__getitem__", &Const{{ util.field_repeated_container_name(field) }}::Get);
    registry.def("Get", &Const{{ util.field_repeated_container_name(field) }}::Get);
{% endif %}
  }
  {
    pybind11::class_<{{ util.field_repeated_container_name(field) }}, std::shared_ptr<{{ util.field_repeated_container_name(field) }}>> registry(m, "{{ util.field_repeated_container_name(field) }}");
    registry.def("__len__", &{{ util.field_repeated_container_name(field) }}::size);
    registry.def("Set", &{{ util.field_repeated_container_name(field) }}::Set);
    registry.def("Clear", &{{ util.field_repeated_container_name(field) }}::Clear);
    registry.def("CopyFrom", (void ({{ util.field_repeated_container_name(field) }}::*)(const Const{{ util.field_repeated_container_name(field) }}&))&{{ util.field_repeated_container_name(field) }}::CopyFrom);
    registry.def("CopyFrom", (void ({{ util.field_repeated_container_name(field) }}::*)(const {{ util.field_repeated_container_name(field) }}&))&{{ util.field_repeated_container_name(field) }}::CopyFrom);
{% if util.field_is_enum_type(field) or util.field_is_message_type(field) %}
    registry.def("Add", (void ({{ util.field_repeated_container_name(field) }}::*)(const {{ util.module_package_cfg_namespace(module) }}::{{ util.field_type_name(field) }}&))&{{ util.field_repeated_container_name(field) }}::Add);
{% else %}
    registry.def("Add", (void ({{ util.field_repeated_container_name(field) }}::*)(const {{ util.field_type_name(field) }}&))&{{ util.field_repeated_container_name(field) }}::Add);
{% endif %}    
    registry.def(pybind11::self == pybind11::self);
    registry.def(pybind11::self < pybind11::self);
{% if util.field_is_message_type(field) %}
    registry.def("__getitem__", (::std::shared_ptr<{{ util.module_package_cfg_namespace(module) }}::{{ util.field_type_name(field) }}> ({{ util.field_repeated_container_name(field) }}::*)(::std::size_t))&{{ util.field_repeated_container_name(field) }}::__SharedMutable__);
    registry.def("Get", (::std::shared_ptr<{{ util.module_package_cfg_namespace(module) }}::{{ util.field_type_name(field) }}> ({{ util.field_repeated_container_name(field) }}::*)(::std::size_t))&{{ util.field_repeated_container_name(field) }}::__SharedMutable__);
    registry.def("Add", &{{ util.field_repeated_container_name(field) }}::__SharedAdd__);
{% else %}
    registry.def("__getitem__", &{{ util.field_repeated_container_name(field) }}::Get);
    registry.def("Get", &{{ util.field_repeated_container_name(field) }}::Get);
    registry.def("__setitem__", &{{ util.field_repeated_container_name(field) }}::Set);
{% endif %}
  }

{# map begin #}
{% elif util.field_has_map_label(field) and util.add_declared_map_field_type_name(field) %}
  {
    pybind11::class_<Const{{ util.field_map_container_name(field) }}, std::shared_ptr<Const{{ util.field_map_container_name(field) }}>> registry(m, "Const{{ util.field_map_container_name(field) }}");
    registry.def("__len__", &Const{{ util.field_map_container_name(field) }}::size);
    registry.def(pybind11::self == pybind11::self);
    registry.def(pybind11::self < pybind11::self);
{% if util.field_is_message_type(util.field_map_value_type(field)) %}
    // lifetime safety is ensured by making iterators for std::pair<const {{ util.field_map_key_type_name(field) }}, std::shared_ptr<Const{{ util.field_map_value_type_name(field) }}>>
    registry.def("__iter__", [](const ::std::shared_ptr<Const{{ util.field_map_container_name(field) }}>& s) { return pybind11::make_iterator(s->shared_const_begin(), s->shared_const_end()); });
    // lifetime safety is ensured by making iterators for std::pair<const {{ util.field_map_key_type_name(field) }}, std::shared_ptr<Const{{ util.field_map_value_type_name(field) }}>>
    registry.def("items", [](const ::std::shared_ptr<Const{{ util.field_map_container_name(field) }}>& s) { return pybind11::make_iterator(s->shared_const_begin(), s->shared_const_end()); });
    registry.def("__getitem__", (::std::shared_ptr<Const{{ util.field_map_value_type_name(field) }}> (Const{{ util.field_map_container_name(field) }}::*)(const {{ util.field_map_key_type_name(field) }}&) const)&Const{{ util.field_map_container_name(field) }}::__SharedConst__);
{% else %}
    registry.def("__iter__", [](const Const{{ util.field_map_container_name(field) }} &s) { return pybind11::make_iterator(s.begin(), s.end()); }, pybind11::keep_alive<0, 1>());
    registry.def("items", [](const Const{{ util.field_map_container_name(field) }} &s) { return pybind11::make_iterator(s.begin(), s.end()); }, pybind11::keep_alive<0, 1>());
    registry.def("__getitem__", &Const{{ util.field_map_container_name(field) }}::Get);
{% endif %}
  }
  {
    pybind11::class_<{{ util.field_map_container_name(field) }}, std::shared_ptr<{{ util.field_map_container_name(field) }}>> registry(m, "{{ util.field_map_container_name(field) }}");
    registry.def("__len__", &{{ util.field_map_container_name(field) }}::size);
    registry.def("Clear", &{{ util.field_map_container_name(field) }}::Clear);
    registry.def("CopyFrom", (void ({{ util.field_map_container_name(field) }}::*)(const Const{{ util.field_map_container_name(field) }}&))&{{ util.field_map_container_name(field) }}::CopyFrom);
    registry.def("CopyFrom", (void ({{ util.field_map_container_name(field) }}::*)(const {{ util.field_map_container_name(field) }}&))&{{ util.field_map_container_name(field) }}::CopyFrom);
    registry.def(pybind11::self == pybind11:: self);
    registry.def(pybind11::self < pybind11:: self);
{% if util.field_is_message_type(util.field_map_value_type(field)) %}
    // lifetime safety is ensured by making iterators for std::pair<const {{ util.field_map_key_type_name(field) }}, std::shared_ptr<{{ util.field_map_value_type_name(field) }}>>
    registry.def("__iter__", [](const ::std::shared_ptr<{{ util.field_map_container_name(field) }}>& s) { return pybind11::make_iterator(s->shared_mut_begin(), s->shared_mut_end()); });
    // lifetime safety is ensured by making iterators for std::pair<const {{ util.field_map_key_type_name(field) }}, std::shared_ptr<{{ util.field_map_value_type_name(field) }}>>
    registry.def("items", [](const ::std::shared_ptr<{{ util.field_map_container_name(field) }}>& s) { return pybind11::make_iterator(s->shared_mut_begin(), s->shared_mut_end()); });
    registry.def("__getitem__", (::std::shared_ptr<{{ util.module_package_cfg_namespace(module) }}::{{ util.field_map_value_type_name(field) }}> ({{ util.field_map_container_name(field) }}::*)(const {{ util.field_map_key_type_name(field) }}&))&{{ util.field_map_container_name(field) }}::__SharedMutable__);
{% else %}
    registry.def("__iter__", [](const {{ util.field_map_container_name(field) }} &s) { return pybind11::make_iterator(s.begin(), s.end()); }, pybind11::keep_alive<0, 1>());
    registry.def("items", [](const {{ util.field_map_container_name(field) }} &s) { return pybind11::make_iterator(s.begin(), s.end()); }, pybind11::keep_alive<0, 1>());
    registry.def("__getitem__", &{{ util.field_map_container_name(field) }}::Get);
    registry.def("__setitem__", &{{ util.field_map_container_name(field) }}::Set);
{% endif %}
  }
{# map end #}

{% endif %}{# field type #}
{% endfor %}{# field #}
{% endif %}{# cls is not entry #}
{% endfor %}{# cls #}
{% for cls in util.module_nested_message_types(module) %}
{% if not util.class_is_map_entry(cls) %}
  {
    pybind11::class_<Const{{ util.class_name(cls) }}, std::shared_ptr<Const{{ util.class_name(cls) }}>> registry(m, "Const{{ util.class_name(cls) }}");
    // the data of `self` will be moved to the result which is always mutable
    registry.def("Move", &Const{{ util.class_name(cls) }}::__Move__);
    registry.def("__id__", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::__Id__);
    registry.def(pybind11::self == pybind11:: self);
    registry.def(pybind11::self < pybind11:: self);
    registry.def("__str__", &Const{{ util.class_name(cls) }}::DebugString);
    registry.def("__repr__", &Const{{ util.class_name(cls) }}::DebugString);
{% for field in util.message_type_fields(cls) %}

{% if util.field_has_required_or_optional_label(field) %}
    registry.def("has_{{ util.field_name(field) }}", &Const{{ util.class_name(cls) }}::has_{{ util.field_name(field) }});
{% if util.field_is_message_type(field) %}
    registry.def("{{ util.field_name(field) }}", &Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
{% else %}
    registry.def("{{ util.field_name(field) }}", &Const{{ util.class_name(cls) }}::{{ util.field_name(field) }});
{% endif %}
{% elif util.field_has_repeated_label(field) %}
    registry.def("{{ util.field_name(field) }}_size", &Const{{ util.class_name(cls) }}::{{ util.field_name(field) }}_size);
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<Const{{ util.field_repeated_container_name(field) }}> (Const{{ util.class_name(cls) }}::*)() const)&Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
{% if util.field_is_message_type(field) %}
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<Const{{ util.field_type_name(field) }}> (Const{{ util.class_name(cls) }}::*)(::std::size_t) const)&Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
{% elif util.field_is_enum_type(field) %}
    registry.def("{{ util.field_name(field) }}", (const {{ util.module_package_cfg_namespace(module) }}::{{ util.field_type_name(field) }}& (Const{{ util.class_name(cls) }}::*)(::std::size_t) const)&Const{{ util.class_name(cls) }}::{{ util.field_name(field) }});
{% else %}
    registry.def("{{ util.field_name(field) }}", (const {{ util.field_type_name(field) }}& (Const{{ util.class_name(cls) }}::*)(::std::size_t) const)&Const{{ util.class_name(cls) }}::{{ util.field_name(field) }});
{% endif %}
{% elif util.field_has_oneof_label(field) %}
    registry.def("has_{{ util.field_name(field) }}", &Const{{ util.class_name(cls) }}::has_{{ util.field_name(field) }});
{% if util.field_is_message_type(field) %}
    registry.def("{{ util.field_name(field) }}", &Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
{% else %}
    registry.def("{{ util.field_name(field) }}", &Const{{ util.class_name(cls) }}::{{ util.field_name(field) }});
{% endif %}{# field message type #}
{# map begin #}
{% elif util.field_has_map_label(field) %}
    registry.def("{{ util.field_name(field) }}_size", &Const{{ util.class_name(cls) }}::{{ util.field_name(field) }}_size);
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<Const{{ util.field_map_container_name(field) }}> (Const{{ util.class_name(cls) }}::*)() const)&Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});

{% if util.field_is_message_type(util.field_map_value_type(field)) %}
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<Const{{ util.field_map_value_type_name(field) }}> (Const{{ util.class_name(cls) }}::*)(const {{ util.field_map_key_type_name(field) }}&) const)&Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
{% elif util.field_is_enum_type(util.field_map_value_type(field)) %}
    registry.def("{{ util.field_name(field) }}", (const {{ util.module_package_cfg_namespace(module) }}::{{ util.field_map_value_type_name(field) }}& (Const{{ util.class_name(cls) }}::*)(const {{ util.field_map_key_type_name(field) }}&) const)&Const{{ util.class_name(cls) }}::{{ util.field_name(field) }});
{% else %}
    registry.def("{{ util.field_name(field) }}", (const {{ util.field_map_value_type_name(field) }}& (Const{{ util.class_name(cls) }}::*)(const {{ util.field_map_key_type_name(field) }}&) const)&Const{{ util.class_name(cls) }}::{{ util.field_name(field) }});
{% endif %}
{# map end #}
{% endif %}{# field label type #}
{% endfor %}{# field #}
{% for oneof in util.message_type_oneofs(cls) %}
    registry.def("{{ util.oneof_name(oneof) }}_case",  &Const{{ util.class_name(cls) }}::{{ util.oneof_name(oneof) }}_case);
    registry.def("has_{{ util.oneof_name(oneof) }}",  &Const{{ util.class_name(cls) }}::has_{{ util.oneof_name(oneof) }});
    registry.def_property_readonly_static("{{ util.oneof_name(oneof).upper() }}_NOT_SET",
        [](const pybind11::object&){ return {{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.oneof_name(oneof).upper() }}_NOT_SET; })
{% for field in util.oneof_type_fields(oneof) %}
        .def_property_readonly_static("{{ util.oneof_type_field_enum_value_name(field) }}", [](const pybind11::object&){ return {{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.oneof_type_field_enum_value_name(field) }}; })
{% endfor %}{# oneof_fields #}
        ;
{% endfor %}{# oneofs #}
  }
  {
    pybind11::class_<{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}, std::shared_ptr<{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}>> registry(m, "{{ util.class_name(cls) }}");
    registry.def(pybind11::init<>());
    registry.def("Clear", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::Clear);
    registry.def("CopyFrom", (void ({{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::*)(const Const{{ util.class_name(cls) }}&))&{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::CopyFrom);
    registry.def("CopyFrom", (void ({{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::*)(const {{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}&))&{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::CopyFrom);
    registry.def("Move", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::__Move__);
    registry.def("__id__", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::__Id__);
    registry.def(pybind11::self == pybind11:: self);
    registry.def(pybind11::self < pybind11:: self);
    registry.def("__str__", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::DebugString);
    registry.def("__repr__", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::DebugString);

{% for oneof in util.message_type_oneofs(cls) %}
    registry.def_property_readonly_static("{{ util.oneof_name(oneof).upper() }}_NOT_SET",
        [](const pybind11::object&){ return {{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.oneof_name(oneof).upper() }}_NOT_SET; })
{% for field in util.oneof_type_fields(oneof) %}
        .def_property_readonly_static("{{ util.oneof_type_field_enum_value_name(field) }}", [](const pybind11::object&){ return {{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.oneof_type_field_enum_value_name(field) }}; })
{% endfor %}{# oneof_fields #}
        ;
{% endfor %}{# oneofs #}

{% for field in util.message_type_fields(cls) %}

{% if util.field_has_required_or_optional_label(field) %}
    registry.def("has_{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::has_{{ util.field_name(field) }});
    registry.def("clear_{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::clear_{{ util.field_name(field) }});
{% if util.field_is_message_type(field) %}
    registry.def("{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
    registry.def("mutable_{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }});
{% else %}
    registry.def("{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.field_name(field) }});
    registry.def("set_{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::set_{{ util.field_name(field) }});
{% endif %}
{% elif util.field_has_repeated_label(field) %}
    registry.def("{{ util.field_name(field) }}_size", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.field_name(field) }}_size);
    registry.def("clear_{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::clear_{{ util.field_name(field) }});
    registry.def("mutable_{{ util.field_name(field) }}", (::std::shared_ptr<{{ util.field_repeated_container_name(field) }}> ({{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::*)())&{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }});
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<Const{{ util.field_repeated_container_name(field) }}> ({{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::*)() const)&{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
{% if util.field_is_message_type(field) %}
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<Const{{ util.field_type_name(field) }}> ({{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::*)(::std::size_t) const)&{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
    registry.def("mutable_{{ util.field_name(field) }}", (::std::shared_ptr<{{ util.module_package_cfg_namespace(module) }}::{{ util.field_type_name(field) }}> ({{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::*)(::std::size_t))&{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }});
{%elif util.field_is_enum_type(field) %}
    registry.def("{{ util.field_name(field) }}", (const {{ util.module_package_cfg_namespace(module) }}::{{ util.field_type_name(field) }}& ({{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::*)(::std::size_t) const)&{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.field_name(field) }});
    registry.def("add_{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::add_{{ util.field_name(field) }});
{% else %}
    registry.def("{{ util.field_name(field) }}", (const {{ util.field_type_name(field) }}& ({{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::*)(::std::size_t) const)&{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.field_name(field) }});
    registry.def("add_{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::add_{{ util.field_name(field) }});
{% endif %}{# field message type #}
{% elif util.field_has_oneof_label(field) %}
    registry.def("has_{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::has_{{ util.field_name(field) }});
    registry.def("clear_{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::clear_{{ util.field_name(field) }});
    registry.def_property_readonly_static("{{ util.oneof_type_field_enum_value_name(field) }}",
        [](const pybind11::object&){ return {{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.oneof_type_field_enum_value_name(field) }}; });
{% if util.field_is_message_type(field) %}
    registry.def("mutable_{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }});
{% else %}
    registry.def("{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.field_name(field) }});
    registry.def("set_{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::set_{{ util.field_name(field) }});
{% endif %}{# field_message_type #}
{# map begin #}
{% elif util.field_has_map_label(field) %}
    registry.def("{{ util.field_name(field) }}_size", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.field_name(field) }}_size);
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<Const{{ util.field_map_container_name(field) }}> ({{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::*)() const)&{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
    registry.def("clear_{{ util.field_name(field) }}", &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::clear_{{ util.field_name(field) }});
    registry.def("mutable_{{ util.field_name(field) }}", (::std::shared_ptr<{{ util.field_map_container_name(field) }}> ({{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::*)())&{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }});
{% if util.field_is_message_type(util.field_map_value_type(field)) %}
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<Const{{ util.field_map_value_type_name(field) }}> ({{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::*)(const {{ util.field_map_key_type_name(field) }}&) const)&{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
{% elif util.field_is_enum_type(util.field_map_value_type(field)) %}
    registry.def("{{ util.field_name(field) }}", (const {{ util.module_package_cfg_namespace(module) }}::{{ util.field_map_value_type_name(field) }}& ({{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::*)(const {{ util.field_map_key_type_name(field) }}&) const)&{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.field_name(field) }});
{% else %}
    registry.def("{{ util.field_name(field) }}", (const {{ util.field_map_value_type_name(field) }}& ({{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::*)(const {{ util.field_map_key_type_name(field) }}&) const)&{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.field_name(field) }});
{% endif %}
{# map end #}
{% endif %}{# field label type #}
{% endfor %}{# field #}
{% for oneof in util.message_type_oneofs(cls) %}
    pybind11::enum_<{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.oneof_enum_name(oneof) }}>(registry, "{{ util.oneof_enum_name(oneof) }}")
        .value("{{ util.oneof_name(oneof).upper() }}_NOT_SET", {{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.oneof_name(oneof).upper() }}_NOT_SET)
{% for field in util.oneof_type_fields(oneof) %}
        .value("{{ util.oneof_type_field_enum_value_name(field) }}", {{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.oneof_type_field_enum_value_name(field) }})
{% endfor %}{# oneof_fields #}
        ;
    registry.def("{{ util.oneof_name(oneof) }}_case",  &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::{{ util.oneof_name(oneof) }}_case);
    registry.def("has_{{ util.oneof_name(oneof) }}",  &{{ util.module_package_cfg_namespace(module) }}::{{ util.class_name(cls) }}::has_{{ util.oneof_name(oneof) }});
{% endfor %}{# oneofs #}
  }
{% endif %}{# cls is not entry #}
{% endfor %}{# cls #}
}
