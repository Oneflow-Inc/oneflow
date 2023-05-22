include(install_flatcc.cmake)

add_subdirectory(attributes)

file(GLOB LITE_SCHEMA_FILES *.fbs)
flatcc_generate(SCHEMA_SRCS ${LITE_SCHEMA_FILES})

add_custom_target(lite_schema_gen DEPENDS ${SCHEMA_SRCS} flatcc-runtime)
add_library(lite_schemas INTERFACE)
add_dependencies(lite_schemas lite_schema_gen lite_attribute_schema_gen)
