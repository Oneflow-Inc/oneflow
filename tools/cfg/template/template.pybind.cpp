#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include "oneflow/cfg/pybind_module_registry.h"
#include "{{ util.module_cfg_header_name(module) }}"

ONEFLOW_CFG_PYBIND11_MODULE("{{ util.module_get_python_module_path(module) }}", m) {
{% if util.module_has_package(module) %}
  using namespace {{ "::".join(util.module_package_list(module)) }}::cfg;
{% else %}
  using namespace cfg;
{% endif %}
{% for enm in util.module_enum_types(module) %}
  {
    pybind11::enum_<{{ util.enum_name(enm) }}> enm(m, "{{ util.enum_name(enm) }}");
{% for value in util.enum_values(enm) %}
    enm.value("{{ util.enum_value_name(value) }}", {{ util.enum_value_name(value) }});
{% endfor %}{# enum_values #}
{% for value in util.enum_values(enm) %}
    m.attr("{{ util.enum_value_name(value) }}") = enm.attr("{{ util.enum_value_name(value) }}");
{% endfor %}{# enum_values #}
  }
{% endfor %}{# enum_types #}
{% for cls in util.module_nested_message_types(module) %}
{% if not util.class_is_map_entry(cls) %}
{% for field in util.message_type_fields(cls) %}
{# no duplicated python class registered for each repeated field type #}
{% if util.field_has_repeated_label(field) and util.add_visited_repeated_field_type_name(field) %}
  if (!ctx->IsTypeIndexRegistered(typeid(_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>)))
  {
    pybind11::class_<_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>, std::shared_ptr<_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>>> registry(m, "_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>");
    registry.def("__len__", &_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::size);
    registry.def(pybind11::self == pybind11:: self);
    registry.def(pybind11::self < pybind11:: self);
{% if util.field_is_message_type(field) %}
    registry.def("__getitem__", (::std::shared_ptr<Const{{ util.field_type_name(field) }}> (_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::*)(::std::size_t) const)&_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::__SharedConst__);
    registry.def("Get", (::std::shared_ptr<Const{{ util.field_type_name(field) }}> (_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::*)(::std::size_t) const)&_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::__SharedConst__);
{% else %}
    registry.def("__getitem__", &_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::Get);
    registry.def("Get", &_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::Get);
{% endif %}
    ctx->RegisterTypeIndex(typeid(_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>));
  }

  if (!ctx->IsTypeIndexRegistered(typeid(_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>)))
  {
    pybind11::class_<_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>, std::shared_ptr<_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>>> registry(m, "_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>");
    registry.def("__len__", &_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::size);
    registry.def("Set", &_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::Set);
    registry.def("Clear", &_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::Clear);
    registry.def("CopyFrom", (void (_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::*)(const _ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>&))&_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::CopyFrom);
    registry.def("CopyFrom", (void (_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::*)(const _RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>&))&_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::CopyFrom);
    registry.def("Add", (void (_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::*)(const {{ util.field_type_name(field) }}&))&_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::Add);
    registry.def(pybind11::self == pybind11:: self);
    registry.def(pybind11::self < pybind11:: self);
{% if util.field_is_message_type(field) %}
    registry.def("__getitem__", (::std::shared_ptr<{{ util.field_type_name(field) }}> (_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::*)(::std::size_t))&_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::__SharedMutable__);
    registry.def("Get", (::std::shared_ptr<{{ util.field_type_name(field) }}> (_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::*)(::std::size_t))&_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::__SharedMutable__);
    registry.def("Add", &_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::__SharedAdd__);
{% else %}
    registry.def("__getitem__", &_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::Get);
    registry.def("Get", &_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::Get);
    registry.def("__setitem__", &_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>::Set);
{% endif %}
    ctx->RegisterTypeIndex(typeid(_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>));
  }


{# map begin #}
{% elif util.field_has_map_label(field) and util.add_visited_map_field_type_name(field) %}
    if (!ctx->IsTypeIndexRegistered(typeid(_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>)))
  {
    pybind11::class_<_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>, std::shared_ptr<_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>>> registry(m, "_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>");
    registry.def("__len__", &_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::size);
    registry.def(pybind11::self == pybind11:: self);
    registry.def(pybind11::self < pybind11:: self);
{% if util.field_is_message_type(util.field_map_value_type(field)) %}
    // lifetime safety is ensured by making iterators for std::pair<const {{ util.field_map_key_type_name(field) }}, std::shared_ptr<Const{{ util.field_map_value_type_name(field) }}>>
    registry.def("__iter__", [](const ::std::shared_ptr<_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>>& s) {
        //return pybind11::make_iterator(s->shared_const_begin(), s->shared_const_end());
        return pybind11::make_iterator(_SharedConstPairIterator_<_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>, Const{{ util.field_map_value_type_name(field) }}>(s->begin()), _SharedConstPairIterator_<_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>, Const{{ util.field_map_value_type_name(field) }}>(s->end()));
        });
    // lifetime safety is ensured by making iterators for std::pair<const {{ util.field_map_key_type_name(field) }}, std::shared_ptr<Const{{ util.field_map_value_type_name(field) }}>>
    registry.def("items", [](const ::std::shared_ptr<_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>>& s) {
        //return pybind11::make_iterator(s->shared_const_begin(), s->shared_const_end());
        return pybind11::make_iterator(_SharedConstPairIterator_<_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>, Const{{ util.field_map_value_type_name(field) }}>(s->begin()), _SharedConstPairIterator_<_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>, Const{{ util.field_map_value_type_name(field) }}>(s->end()));
        });
    registry.def("__getitem__", (::std::shared_ptr<Const{{ util.field_map_value_type_name(field) }}> (_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::*)(const {{ util.field_map_key_type_name(field) }}&) const)&_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::__SharedConst__);
{% else %}
    registry.def("__iter__", [](const _ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}> &s) { return pybind11::make_iterator(s.begin(), s.end()); }, pybind11::keep_alive<0, 1>());
    registry.def("items", [](const _ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}> &s) { return pybind11::make_iterator(s.begin(), s.end()); }, pybind11::keep_alive<0, 1>());
    registry.def("__getitem__", &_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::Get);
{% endif %}
    ctx->RegisterTypeIndex(typeid(_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>));
  }
  if (!ctx->IsTypeIndexRegistered(typeid(_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>)))
  {
    pybind11::class_<_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>, std::shared_ptr<_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>>> registry(m, "_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>");
    registry.def("__len__", &_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::size);
    registry.def("Clear", &_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::Clear);
    registry.def("CopyFrom", (void (_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::*)(const _ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>&))&_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::CopyFrom);
    registry.def("CopyFrom", (void (_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::*)(const _MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>&))&_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::CopyFrom);
    registry.def(pybind11::self == pybind11:: self);
    registry.def(pybind11::self < pybind11:: self);
{% if util.field_is_message_type(util.field_map_value_type(field)) %}
    // lifetime safety is ensured by making iterators for std::pair<const {{ util.field_map_key_type_name(field) }}, std::shared_ptr<{{ util.field_map_value_type_name(field) }}>>
    registry.def("__iter__", [](const ::std::shared_ptr<_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>>& s) { return pybind11::make_iterator(s->shared_mut_begin(), s->shared_mut_end()); });
    // lifetime safety is ensured by making iterators for std::pair<const {{ util.field_map_key_type_name(field) }}, std::shared_ptr<{{ util.field_map_value_type_name(field) }}>>
    registry.def("items", [](const ::std::shared_ptr<_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>>& s) { return pybind11::make_iterator(s->shared_mut_begin(), s->shared_mut_end()); });
    registry.def("__getitem__", (::std::shared_ptr<{{ util.field_map_value_type_name(field) }}> (_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::*)(const {{ util.field_map_key_type_name(field) }}&))&_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::__SharedMutable__);
{% else %}
    registry.def("__iter__", [](const _MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}> &s) { return pybind11::make_iterator(s.begin(), s.end()); }, pybind11::keep_alive<0, 1>());
    registry.def("items", [](const _MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}> &s) { return pybind11::make_iterator(s.begin(), s.end()); }, pybind11::keep_alive<0, 1>());
    registry.def("__getitem__", &_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::Get);
    registry.def("__setitem__", &_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>::Set);
{% endif %}
    ctx->RegisterTypeIndex(typeid(_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>));
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
    registry.def("__id__", &{{ util.class_name(cls) }}::__Id__);
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
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>> (Const{{ util.class_name(cls) }}::*)() const)&Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
{% if util.field_is_message_type(field) %}
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<Const{{ util.field_type_name(field) }}> (Const{{ util.class_name(cls) }}::*)(::std::size_t) const)&Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
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
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>> (Const{{ util.class_name(cls) }}::*)() const)&Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});

{% if util.field_is_message_type(util.field_map_value_type(field)) %}
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<Const{{ util.field_map_value_type_name(field) }}> (Const{{ util.class_name(cls) }}::*)(const {{ util.field_map_key_type_name(field) }}&) const)&Const{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
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
        [](const pybind11::object&){ return {{ util.class_name(cls) }}::{{ util.oneof_name(oneof).upper() }}_NOT_SET; })
{% for field in util.oneof_type_fields(oneof) %}
        .def_property_readonly_static("{{ util.oneof_type_field_enum_value_name(field) }}", [](const pybind11::object&){ return {{ util.class_name(cls) }}::{{ util.oneof_type_field_enum_value_name(field) }}; })
{% endfor %}{# oneof_fields #}
        ;
{% endfor %}{# oneofs #}
  }
  {
    pybind11::class_<{{ util.class_name(cls) }}, std::shared_ptr<{{ util.class_name(cls) }}>> registry(m, "{{ util.class_name(cls) }}");
    registry.def(pybind11::init<>());
    registry.def("Clear", &{{ util.class_name(cls) }}::Clear);
    registry.def("CopyFrom", (void ({{ util.class_name(cls) }}::*)(const Const{{ util.class_name(cls) }}&))&{{ util.class_name(cls) }}::CopyFrom);
    registry.def("CopyFrom", (void ({{ util.class_name(cls) }}::*)(const {{ util.class_name(cls) }}&))&{{ util.class_name(cls) }}::CopyFrom);
    registry.def("Move", &{{ util.class_name(cls) }}::__Move__);
    registry.def("__id__", &{{ util.class_name(cls) }}::__Id__);
    registry.def(pybind11::self == pybind11:: self);
    registry.def(pybind11::self < pybind11:: self);
    registry.def("__str__", &{{ util.class_name(cls) }}::DebugString);
    registry.def("__repr__", &{{ util.class_name(cls) }}::DebugString);

{% for oneof in util.message_type_oneofs(cls) %}
    registry.def_property_readonly_static("{{ util.oneof_name(oneof).upper() }}_NOT_SET",
        [](const pybind11::object&){ return {{ util.class_name(cls) }}::{{ util.oneof_name(oneof).upper() }}_NOT_SET; })
{% for field in util.oneof_type_fields(oneof) %}
        .def_property_readonly_static("{{ util.oneof_type_field_enum_value_name(field) }}", [](const pybind11::object&){ return {{ util.class_name(cls) }}::{{ util.oneof_type_field_enum_value_name(field) }}; })
{% endfor %}{# oneof_fields #}
        ;
{% endfor %}{# oneofs #}

{% for field in util.message_type_fields(cls) %}

{% if util.field_has_required_or_optional_label(field) %}
    registry.def("has_{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::has_{{ util.field_name(field) }});
    registry.def("clear_{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::clear_{{ util.field_name(field) }});
{% if util.field_is_message_type(field) %}
    registry.def("{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
    registry.def("mutable_{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }});
{% else %}
    registry.def("{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::{{ util.field_name(field) }});
    registry.def("set_{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::set_{{ util.field_name(field) }});
{% endif %}
{% elif util.field_has_repeated_label(field) %}
    registry.def("{{ util.field_name(field) }}_size", &{{ util.class_name(cls) }}::{{ util.field_name(field) }}_size);
    registry.def("clear_{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::clear_{{ util.field_name(field) }});
    registry.def("mutable_{{ util.field_name(field) }}", (::std::shared_ptr<_RepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>> ({{ util.class_name(cls) }}::*)())&{{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }});
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<_ConstRepeatedField_<{{ util.field_type_name_with_cfg_namespace(field) }}>> ({{ util.class_name(cls) }}::*)() const)&{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
{% if util.field_is_message_type(field) %}
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<Const{{ util.field_type_name(field) }}> ({{ util.class_name(cls) }}::*)(::std::size_t) const)&{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
    registry.def("mutable_{{ util.field_name(field) }}", (::std::shared_ptr<{{ util.field_type_name(field) }}> ({{ util.class_name(cls) }}::*)(::std::size_t))&{{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }});
{% else %}
    registry.def("{{ util.field_name(field) }}", (const {{ util.field_type_name(field) }}& ({{ util.class_name(cls) }}::*)(::std::size_t) const)&{{ util.class_name(cls) }}::{{ util.field_name(field) }});
    registry.def("add_{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::add_{{ util.field_name(field) }});
{% endif %}{# field message type #}
{% elif util.field_has_oneof_label(field) %}
    registry.def("has_{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::has_{{ util.field_name(field) }});
    registry.def("clear_{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::clear_{{ util.field_name(field) }});
    registry.def_property_readonly_static("{{ util.oneof_type_field_enum_value_name(field) }}",
        [](const pybind11::object&){ return {{ util.class_name(cls) }}::{{ util.oneof_type_field_enum_value_name(field) }}; });
{% if util.field_is_message_type(field) %}
    registry.def("mutable_{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }});
{% else %}
    registry.def("{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::{{ util.field_name(field) }});
    registry.def("set_{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::set_{{ util.field_name(field) }});
{% endif %}{# field_message_type #}
{# map begin #}
{% elif util.field_has_map_label(field) %}
    registry.def("{{ util.field_name(field) }}_size", &{{ util.class_name(cls) }}::{{ util.field_name(field) }}_size);
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<_ConstMapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>> ({{ util.class_name(cls) }}::*)() const)&{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
    registry.def("clear_{{ util.field_name(field) }}", &{{ util.class_name(cls) }}::clear_{{ util.field_name(field) }});
    registry.def("mutable_{{ util.field_name(field) }}", (::std::shared_ptr<_MapField_<{{ util.field_map_pair_type_name_with_cfg_namespace(field) }}>> ({{ util.class_name(cls) }}::*)())&{{ util.class_name(cls) }}::shared_mutable_{{ util.field_name(field) }});
{% if util.field_is_message_type(util.field_map_value_type(field)) %}
    registry.def("{{ util.field_name(field) }}", (::std::shared_ptr<Const{{ util.field_map_value_type_name(field) }}> ({{ util.class_name(cls) }}::*)(const {{ util.field_map_key_type_name(field) }}&) const)&{{ util.class_name(cls) }}::shared_const_{{ util.field_name(field) }});
{% else %}
    registry.def("{{ util.field_name(field) }}", (const {{ util.field_map_value_type_name(field) }}& ({{ util.class_name(cls) }}::*)(const {{ util.field_map_key_type_name(field) }}&) const)&{{ util.class_name(cls) }}::{{ util.field_name(field) }});
{% endif %}
{# map end #}
{% endif %}{# field label type #}
{% endfor %}{# field #}
{% for oneof in util.message_type_oneofs(cls) %}
    pybind11::enum_<{{ util.class_name(cls) }}::{{ util.oneof_enum_name(oneof) }}>(registry, "{{ util.oneof_enum_name(oneof) }}")
        .value("{{ util.oneof_name(oneof).upper() }}_NOT_SET", {{ util.class_name(cls) }}::{{ util.oneof_name(oneof).upper() }}_NOT_SET)
{% for field in util.oneof_type_fields(oneof) %}
        .value("{{ util.oneof_type_field_enum_value_name(field) }}", {{ util.class_name(cls) }}::{{ util.oneof_type_field_enum_value_name(field) }})
{% endfor %}{# oneof_fields #}
        ;
    registry.def("{{ util.oneof_name(oneof) }}_case",  &{{ util.class_name(cls) }}::{{ util.oneof_name(oneof) }}_case);
    registry.def("has_{{ util.oneof_name(oneof) }}",  &{{ util.class_name(cls) }}::has_{{ util.oneof_name(oneof) }});
{% endfor %}{# oneofs #}
  }
{% endif %}{# cls is not entry #}
{% endfor %}{# cls #}
}
