import re

class ProtoReflectionUtil:
    def __init__(self):
        self.visited_repeated_field_type_name_ = set()
        self.visited_map_field_type_name_ = set()

    def module_dependencies(self, module):
        return module.dependencies

    def module_has_package(self, module):
        return module.package != ""

    def module_package_list(self, module):
        return filter(lambda x: len(x) > 0, module.package.split("."))

    def module_cfg_header_name(self, module):
        return module.name[0:-5] + "cfg.h"

    def module_proto_header_name(self, module):
        return module.name[0:-5] + "pb.h"

    def module_cfg_convert_header_name(self, module):
        return module.name[0:-5] + "cfg.proto.convert.h"
    
    def module_get_python_module_path(self, module):
        return module.name[0:-6].replace('/', '.')

    def module_header_macro_lock(self, module):
        return _ToValidVarName("CFG_%s_"% self.module_cfg_header_name(module).upper())

    def module_proto_convert_header_macro_lock(self, module):
        return _ToValidVarName("CFG_%s_"% self.module_cfg_convert_header_name(module).upper())

    def module_enum_types(self, module):
        return module.enum_types_by_name.values()

    def module_message_types(self, module):
        return module.message_types_by_name.values()

    def enum_name(self, enum):
        return enum.name

    def enum_values(self, enum):
        return enum.values

    def enum_value_name(self, enum_value):
        return enum_value.name

    def enum_value_number(self, enum_value):
        return enum_value.number

    def message_type_fields(self, cls):
        return cls.fields

    def cls_has_oneofs(self, cls):
        return self.cls_oneofs_size(cls) != 0

    def cls_oneofs_size(self, cls):
        return len(cls.oneofs)

    def message_type_oneofs(self, cls):
        return cls.oneofs

    def oneof_name(self, oneof):
        return oneof.name

    def oneof_enum_name(self, oneof):
        return self.oneof_camel_name(oneof) + "Case"
    
    def oneof_camel_name(self, oneof):
        return self._underline_name_to_camel(oneof.name)

    def oneof_name_of_oneof_type_field(self, field):
        return field.containing_oneof.name

    def oneof_type_fields(self, oneof):
        return oneof.fields

    def oneof_type_field_name(self, field):
        return field.name

    def oneof_type_field_enum_value_name(self, field):
        return 'k' + self._underline_name_to_camel(field.name)

    def oneof_type_field_enum_value_number(self, field):
        return field.number

    def field_has_oneof_label(self, field):
        return field.containing_oneof is not None
    
    def field_oneof_name(self, field):
        assert self.field_has_oneof_label(field)
        return field.containing_oneof.name
    
    def field_has_required_label(self, field):
        return field.label == field.LABEL_REQUIRED

    def field_has_optional_label(self, field):
        return field.label == field.LABEL_OPTIONAL and not self.field_in_oneof(field)

    def field_has_required_or_optional_label(self, field):
        return self.field_has_required_label(field) or self.field_has_optional_label(field)

    def field_has_repeated_label(self, field):
        return field.label == field.LABEL_REPEATED and not self._field_is_map_entry(field)

    def field_is_map(self, field):
        return field.label == field.LABEL_REPEATED and self._field_is_map_entry(field)

    def field_in_oneof(self, field):
        return field.containing_oneof is not None

    def field_has_default_value(self, field):
        return field.has_default_value

    def field_default_value_literal(self, field):
        if field.cpp_type == field.CPPTYPE_STRING:
            return '"%s"' % field.default_value
        return field.default_value

    def field_name(self, field):
        return field.name

    def field_type_name(self, field):
        if self.field_is_message_type(field):
            return self.field_message_type_name(field)
        return self.field_scalar_type_name(field)
    
    def field_map_key_type_name(self, field):
        return self.field_type_name(field.message_type.fields_by_name['key'])

    def field_map_value_type_name(self, field):
        return self.field_type_name(field.message_type.fields_by_name['value'])

    def field_map_value_type_is_message(self, field):
        return self.field_is_message_type(field.message_type.fields_by_name['value'])

    def field_map_value_type_is_enum(self, field):
        return self.field_is_enum_type(field.message_type.fields_by_name['value'])
    
    def field_map_value_type_enum_name(self, field):
        return self.field_enum_name(field.message_type.fields_by_name['value'])
    
    def field_map_value_type(self, field):
        return field.message_type.fields_by_name['value']

    def field_map_pair_type_name(self, field):
        return f'{self.field_map_key_type_name(field)}, {self.field_map_value_type_name(field)}'

    def field_is_message_type(self, field):
        return field.message_type is not None

    def field_message_type_name(self, field):
        return field.message_type.name

    def field_repeated_container_name(self, field):
        module_prefix = self.module_header_macro_lock(field.containing_type.file)
        type_name = self.field_type_name(field)
        return _ToValidVarName("_%s_RepeatedField_%s_"%(module_prefix, type_name))

    def field_map_pair_type_name_with_underline(self, field):
        return f'{self.field_map_key_type_name(field)}_{self.field_map_value_type_name(field)}'

    def field_map_container_name(self, field):
        module_prefix = self.module_header_macro_lock(field.containing_type.file)
        type_name = self.field_map_pair_type_name_with_underline(field)
        return _ToValidVarName("_%s_MapField_%s_"%(module_prefix, type_name))

    def field_is_enum_type(self, field):
        return field.enum_type is not None

    def field_enum_name(self, field):
        return field.enum_type.name

    def field_scalar_type_name(self, field):
        if field.cpp_type == field.CPPTYPE_BOOL:
            return "bool"
        if field.cpp_type == field.CPPTYPE_ENUM:
            return field.enum_type.name
        if field.cpp_type == field.CPPTYPE_DOUBLE:
            return "double"
        if field.cpp_type == field.CPPTYPE_FLOAT:
            return "float"
        if field.cpp_type == field.CPPTYPE_INT32:
            return "int32_t"
        if field.cpp_type == field.CPPTYPE_INT64:
            return "int64_t"
        if field.cpp_type == field.CPPTYPE_INT64:
            return "int64_t"
        if field.cpp_type == field.CPPTYPE_STRING:
            return "::std::string"
        if field.cpp_type == field.CPPTYPE_UINT32:
            return "uint32_t"
        if field.cpp_type == field.CPPTYPE_UINT64:
            return "uint64_t"
        raise NotImplementedError("field.cpp_type is %s"%field.cpp_type)

    # return True if added first time
    def add_visited_repeated_field_type_name(self, field):
        field_type_name = self.field_type_name(field)
        if field_type_name in self.visited_repeated_field_type_name_:
            return False
        self.visited_repeated_field_type_name_.add(field_type_name)
        return True

    # return True if added first time
    def add_visited_map_field_type_name(self, field):
        field_map_pair_type_name = self.field_map_pair_type_name(field)
        if field_map_pair_type_name in self.visited_map_field_type_name_:
            return False
        self.visited_map_field_type_name_.add(field_map_pair_type_name)
        return True

    def _field_is_map_entry(self, field):
        if field.message_type is None:
            return False
        capitalized_name = field.camelcase_name[0].capitalize() + field.camelcase_name[1:]
        entry_type_name = capitalized_name + "Entry"
        if field.message_type.name != entry_type_name:
            return False
        entry_fields = field.message_type.fields
        if len(entry_fields) != 2:
            return False
        if entry_fields[0].name != 'key':
            return False
        if entry_fields[1].name != 'value':
            return False
        return True
    
    def _underline_name_to_camel(self, name):
        sub_name_list = name.split('_')
        camel_name = ''
        for sub_name in sub_name_list:
            camel_name = camel_name + sub_name[0].upper() + sub_name[1:]
        return camel_name

def _ToValidVarName(s):
    return re.sub("[^a-zA-Z0-9]", "_", s)
