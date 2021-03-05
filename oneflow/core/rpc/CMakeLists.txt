add_library(of_rpc
    lib/local.cpp
)
add_dependencies(of_rpc of_cfgobj)
add_dependencies(of_rpc of_protoobj)
