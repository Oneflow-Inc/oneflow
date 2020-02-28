#ifndef ONEFLOW_CORE_EXTENSION_EXT_REGISTRATION_H
#define ONEFLOW_CORE_EXTENSION_EXT_REGISTRATION_H
#include "oneflow/core/extension/extension_registration.h"

#define REGISTER_EXTENSION(event_name, ext_constructor)                        \
  static ::oneflow::extension::Registrar OF_PP_CAT(g_registrar, __COUNTER__) = \
      ::oneflow::extension::Registrar(event_name, ext_constructor)

#endif  // ONEFLOW_CORE_EXTENSION_EXT_REGISTRATION_H
