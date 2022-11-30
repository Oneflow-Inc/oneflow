file(GLOB LITE_ATTRIBUTE_SCHEMA_FILES *.fbs)
flatcc_generate(ATTRIBUTE_SCHEMA_SRCS ${LITE_ATTRIBUTE_SCHEMA_FILES})

add_custom_target(lite_attribute_schema_gen DEPENDS ${ATTRIBUTE_SCHEMA_SRCS} flatcc-runtime)
