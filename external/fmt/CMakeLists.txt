include(FetchContent)
FetchContent_Declare(fmt URL ${FMT_URL} URL_HASH MD5=${FMT_MD5})

FetchContent_MakeAvailable(fmt)

install(TARGETS fmt EXPORT oneflow DESTINATION lib)