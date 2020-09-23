#ifndef {{ util.module_proto_convert_header_macro_lock(module) }}
#define {{ util.module_proto_convert_header_macro_lock(module) }}

#include "{{ util.module_cfg_header_name(module) }}"
#include "{{ util.module_proto_header_name(module) }}"

namespace oneflow {

using cfg = ::oneflow::cfg;

{% for enm in util.module_enum_types(module) %}

::oneflow::{{ util.enum_name(enm) }} Cfg{{ util.enum_name(enm) }}ToProto{{ util.enum_name(enm) }}(const cfg::{{ util.enum_name(enm) }}& cfg_enum);

cfg::{{ util.enum_name(enm) }} Proto{{ util.enum_name(enm) }}ToCfg{{ util.enum_name(enm) }}(const ::oneflow::{{ util.enum_name(enm) }}& proto_enum);
{% endfor %}{# enms #}

{% for cls in util.module_message_types(module) %}

cfg::{{ util.class_name(cls) }} FromProto(const ::oneflow::{{ util.class_name(cls) }}& proto_{{ util.class_name(cls).lower() }});

::oneflow::{{ util.class_name(cls) }} ToProto(const cfg::{{ util.class_name(cls) }}& cfg_{{ util.class_name(cls).lower() }});
{% endfor %}{# cls #}

} // namespace oneflow

#endif  // {{ util.module_proto_convert_header_macro_lock(module) }}
