import re


class ProtoReflectionUtil:
    def __init__(self):
        self.declared_repeated_field_type_name_ = set()
        self.declared_map_field_type_name_ = set()
        self.defined_repeated_field_type_name_ = set()
        self.defined_map_field_type_name_ = set()

    def module_dependencies(self, module):
        return module.dependencies

    def module_has_package(self, module):
        return module.package != ""

    def module_package_list(self, module):
        return filter(lambda x: len(x) > 0, module.package.split("."))

    def module_package_namespace(self, module):
        if module.package:
            return "::" + module.package.replace(".", "::")
        else:
            return ""

    def module_package_cfg_namespace(self, module):
        return self.module_package_namespace(module) + "::cfg"

    def module_cfg_header_name(self, module):
        return module.name[0:-5] + "cfg.h"

    def module_proto_header_name(self, module):
        return module.name[0:-5] + "pb.h"

    def module_get_python_module_path(self, module):
        return module.name[0:-6].replace("/", ".")

    def module_header_macro_lock(self, module):
        return _ToValidVarName("CFG_%s_" % self.module_cfg_header_name(module).upper())

    def module_enum_types(self, module):
        return module.enum_types_by_name.values()

    def module_nested_message_types(self, module):
        def FlattenMessageTypes(message_types):
            for msg_type in message_types:
                for nested_msg_type in FlattenMessageTypes(msg_type.nested_types):
                    yield nested_msg_type
                yield msg_type

        return [
            msg_type
            for msg_type in FlattenMessageTypes(module.message_types_by_name.values())
        ]

    def class_name(self, cls):
        package = cls.file.package
        if package:
            return (cls.full_name)[len(package) + 1 :].replace(".", "_")
        else:
            return (cls.full_name).replace(".", "_")

    def class_name_under_line(self, cls):
        return "_" + self.class_name(cls) + "_"

    def class_name_const(self, cls):
        return "Const" + self.class_name(cls)

    def class_is_map_entry(self, cls):
        return self.class_name(cls).endswith("Entry")

    def enum_name(self, enum):
        package = enum.file.package
        if package:
            return (enum.full_name)[len(package) + 1 :].replace(".", "_")
        else:
            return (enum.full_name).replace(".", "_")

    def enum_values(self, enum):
        return enum.values

    def enum_value_name(self, enum_value):
        return enum_value.name

    def enum_value_number(self, enum_value):
        return enum_value.number

    def message_type_fields(self, cls):
        return cls.fields

    def has_message_type_fields(self, cls):
        return len(self.message_type_fields(cls)) > 0

    def message_type_oneofs(self, cls):
        return cls.oneofs

    def message_type_enums(self, cls):
        return cls.enum_types

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

    def oneof_type_field_enum_value_name(self, field):
        return "k" + self._underline_name_to_camel(field.name)

    def field_number(self, field):
        return field.number

    def field_oneof_name(self, field):
        assert self.field_has_oneof_label(field)
        return field.containing_oneof.name

    def field_oneof_struct_name(self, field):
        assert self.field_has_oneof_label(field)
        return self._underline_name_to_camel(field.containing_oneof.name) + "Struct"

    def field_has_required_label(self, field):
        return field.label == field.LABEL_REQUIRED

    def field_has_optional_label(self, field):
        return field.label == field.LABEL_OPTIONAL and not self.field_in_oneof(field)

    def field_has_required_or_optional_label(self, field):
        return self.field_has_required_label(field) or self.field_has_optional_label(
            field
        )

    def field_has_repeated_label(self, field):
        return field.label == field.LABEL_REPEATED and not self._field_is_map_entry(
            field
        )

    def field_has_oneof_label(self, field):
        return field.containing_oneof is not None

    def field_has_map_label(self, field):
        return field.label == field.LABEL_REPEATED and self._field_is_map_entry(field)

    def field_in_oneof(self, field):
        return field.containing_oneof is not None

    def field_has_default_value(self, field):
        return field.has_default_value

    def field_default_value_literal(self, field):
        if field.cpp_type == field.CPPTYPE_STRING:
            return '"%s"' % field.default_value
        if field.cpp_type == field.CPPTYPE_BOOL:
            return ("%s" % field.default_value).lower()
        return field.default_value

    def field_name(self, field):
        return field.name

    def field_type_name(self, field):
        if self.field_is_message_type(field):
            return self.field_message_type_name(field)
        return self.field_scalar_type_name(field)

    def field_type_name_const_with_cfg_namespace(self, field):
        if self.field_is_message_type(field):
            return self.field_message_type_const_name_with_cfg_namespace(field)
        elif self.field_is_enum_type(field):
            return self.field_enum_name_with_cfg_namespace(field)
        return self.field_scalar_type_name(field)

    def field_type_name_with_cfg_namespace(self, field):
        if self.field_is_message_type(field):
            return self.field_message_type_name_with_cfg_namespace(field)
        elif self.field_is_enum_type(field):
            return self.field_enum_name_with_cfg_namespace(field)
        return self.field_scalar_type_name(field)

    def field_map_key_type_name(self, field):
        return self.field_type_name(field.message_type.fields_by_name["key"])

    def field_map_value_type_name(self, field):
        return self.field_type_name(field.message_type.fields_by_name["value"])

    def field_map_value_type_name_with_cfg_namespace(self, field):
        return self.field_type_name_with_cfg_namespace(
            field.message_type.fields_by_name["value"]
        )

    def field_map_value_type_is_message(self, field):
        return self.field_is_message_type(field.message_type.fields_by_name["value"])

    def field_map_value_type_is_enum(self, field):
        return self.field_is_enum_type(field.message_type.fields_by_name["value"])

    def field_map_value_type_enum_name(self, field):
        return self.field_enum_name(field.message_type.fields_by_name["value"])

    def field_map_value_type(self, field):
        return field.message_type.fields_by_name["value"]

    def field_map_pair_type_name(self, field):
        return "%s, %s" % (
            self.field_map_key_type_name(field),
            self.field_map_value_type_name(field),
        )

    def field_map_pair_type_name_with_cfg_namespace(self, field):
        return "%s, %s" % (
            self.field_map_key_type_name(field),
            self.field_map_value_type_name_with_cfg_namespace(field),
        )

    def field_is_message_type(self, field):
        return field.message_type is not None

    def field_message_type_name(self, field):
        package = field.message_type.file.package
        if package:
            return (field.message_type.full_name)[len(package) + 1 :].replace(".", "_")
        else:
            return (field.message_type.full_name).replace(".", "_")

    def other_file_declared_namespaces_and_enum_name(self, module):
        enum_defined_in_this_module = set()
        for cls in self.module_nested_message_types(module):
            enum_defined_in_this_module.add(cls)
        enum_defined_in_other_module = set()

        def TryAddEnum(field):
            if (
                field.enum_type is not None
                and field.enum_type not in enum_defined_in_this_module
            ):
                enum_defined_in_other_module.add(field.enum_type)

        for cls in self.module_nested_message_types(module):
            for field in self.message_type_fields(cls):
                if self.field_has_map_label(field):
                    TryAddEnum(field.message_type.fields_by_name["key"])
                    TryAddEnum(field.message_type.fields_by_name["value"])
                else:
                    TryAddEnum(field)
        namespaces = []
        enums = []
        for enum in enum_defined_in_other_module:
            package = enum.file.package
            if package != "":
                namespaces.append(package.split("."))
                enums.append(enum.full_name[len(package) + 1 :].replace(".", "_"))
            else:
                namespaces.append([])
                enums.append(enum.full_name.replace(".", "_"))
        return sorted(zip(namespaces, enums))

    def other_file_declared_namespaces_and_class_name(self, module):
        cls_defined_in_this_module = set()
        for cls in self.module_nested_message_types(module):
            cls_defined_in_this_module.add(cls)
        cls_defined_in_other_module = set()

        def TryAddClass(field):
            if (
                field.message_type is not None
                and field.message_type not in cls_defined_in_this_module
            ):
                cls_defined_in_other_module.add(field.message_type)

        for cls in self.module_nested_message_types(module):
            for field in self.message_type_fields(cls):
                if self.field_has_map_label(field):
                    TryAddClass(field.message_type.fields_by_name["value"])
                else:
                    TryAddClass(field)
        namespaces = []
        clss = []
        for cls in cls_defined_in_other_module:
            package = cls.file.package
            if package != "":
                namespaces.append(package.split("."))
                clss.append(cls.full_name[len(package) + 1 :].replace(".", "_"))
            else:
                namespaces.append([])
                clss.append(enum.full_name.replace(".", "_"))
        return sorted(zip(namespaces, clss))

    def field_message_type_name_with_cfg_namespace(self, field):
        package = field.message_type.file.package
        if package:
            return "::%s::cfg::%s" % (
                package.replace(".", "::"),
                (field.message_type.full_name)[len(package) + 1 :].replace(".", "_"),
            )
        else:
            return "::cfg::%s" % ((field.message_type.full_name).replace(".", "_"))

    def field_message_type_const_name_with_cfg_namespace(self, field):
        package = field.message_type.file.package
        if package:
            return "::%s::cfg::%s%s" % (
                package.replace(".", "::"),
                "Const",
                (field.message_type.full_name)[len(package) + 1 :].replace(".", "_"),
            )
        else:
            return "::cfg::%s%s" % (
                "Const",
                (field.message_type.full_name).replace(".", "_"),
            )

    def field_message_type_name_with_proto_namespace(self, field):
        package = field.message_type.file.package
        if package:
            return "::%s::%s" % (
                package.replace(".", "::"),
                (field.message_type.full_name)[len(package) + 1 :].replace(".", "_"),
            )
        else:
            return "::%s" % ((field.message_type.full_name).replace(".", "_"))

    def field_repeated_container_name(self, field):
        module_prefix = self.module_header_macro_lock(field.containing_type.file)
        type_name = self.field_type_name(field)
        return _ToValidVarName("_%s_RepeatedField_%s_" % (module_prefix, type_name))

    def field_map_pair_type_name_with_underline(self, field):
        return "%s_%s" % (
            self.field_map_key_type_name(field),
            self.field_map_value_type_name(field),
        )

    def field_map_container_name(self, field):
        module_prefix = self.module_header_macro_lock(field.containing_type.file)
        type_name = self.field_map_pair_type_name_with_underline(field)
        return _ToValidVarName("_%s_MapField_%s_" % (module_prefix, type_name))

    def field_is_enum_type(self, field):
        return field.enum_type is not None

    def field_enum_name(self, field):
        return self.enum_name(field.enum_type)

    def field_enum_cfg_namespace(self, field):
        if self.field_has_map_label(field):
            package = field.message_type.fields_by_name["value"].enum_type.file.package
        else:
            package = field.enum_type.file.package

        if package:
            return "::" + package.replace(".", "::") + "::cfg"
        else:
            return "::cfg"

    def field_enum_name_with_cfg_namespace(self, field):
        if self.field_has_map_label(field):
            field = field.message_type.fields_by_name["value"]
        package = field.enum_type.file.package
        if package != "":
            return (
                "::"
                + package.replace(".", "::")
                + "::cfg::"
                + self.enum_name(field.enum_type)
            )
        else:
            return "::cfg::" + self.enum_name(field.enum_type)

    def field_enum_name_with_proto_namespace(self, field):
        if self.field_has_map_label(field):
            field = field.message_type.fields_by_name["value"]
        package = field.enum_type.file.package
        if package != "":
            return (
                "::"
                + package.replace(".", "::")
                + "::"
                + self.enum_name(field.enum_type)
            )
        else:
            return "::" + self.enum_name(field.enum_type)

    def field_is_string_type(self, field):
        return self.field_type_name(field) == "::std::string"

    def field_scalar_type_name(self, field):
        if field.cpp_type == field.CPPTYPE_BOOL:
            return "bool"
        if field.cpp_type == field.CPPTYPE_ENUM:
            return self.enum_name(field.enum_type)
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
        raise NotImplementedError("field.cpp_type is %s" % field.cpp_type)

    # return True if added first time
    def add_declared_repeated_field_type_name(self, field):
        field_type_name = self.field_type_name(field)
        if field_type_name in self.declared_repeated_field_type_name_:
            return False
        self.declared_repeated_field_type_name_.add(field_type_name)
        return True

    # return True if added first time
    def add_declared_map_field_type_name(self, field):
        field_map_pair_type_name = self.field_map_pair_type_name(field)
        if field_map_pair_type_name in self.declared_map_field_type_name_:
            return False
        self.declared_map_field_type_name_.add(field_map_pair_type_name)
        return True

    # return True if added first time
    def add_defined_repeated_field_type_name(self, field):
        field_type_name = self.field_type_name(field)
        if field_type_name in self.defined_repeated_field_type_name_:
            return False
        self.defined_repeated_field_type_name_.add(field_type_name)
        return True

    # return True if added first time
    def add_defined_map_field_type_name(self, field):
        field_map_pair_type_name = self.field_map_pair_type_name(field)
        if field_map_pair_type_name in self.defined_map_field_type_name_:
            return False
        self.defined_map_field_type_name_.add(field_map_pair_type_name)
        return True

    def _field_is_map_entry(self, field):
        if field.message_type is None:
            return False
        capitalized_name = (
            field.camelcase_name[0].capitalize() + field.camelcase_name[1:]
        )
        entry_type_name = capitalized_name + "Entry"
        if field.message_type.name != entry_type_name:
            return False
        entry_fields = field.message_type.fields
        if len(entry_fields) != 2:
            return False
        if entry_fields[0].name != "key":
            return False
        if entry_fields[1].name != "value":
            return False
        return True

    def _underline_name_to_camel(self, name):
        sub_name_list = name.split("_")
        camel_name = ""
        for sub_name in sub_name_list:
            camel_name = camel_name + sub_name[0].upper() + sub_name[1:]

        flag = False
        length = len(camel_name)
        for i in range(length):
            if flag and camel_name[i].isalpha():
                camel_name = (
                    camel_name[0:i] + camel_name[i].upper() + camel_name[i + 1 :]
                )
                flag = False
            if camel_name[i].isdigit():
                flag = True
        return camel_name


def _ToValidVarName(s):
    return re.sub("[^a-zA-Z0-9]", "_", s)
