file(GLOB_RECURSE ONEFLOW_GRAPH_SRCS "*.h")
file(GLOB_RECURSE ONEFLOW_GRAPH_HDRS "*.cpp")
add_library(of_graph
    ${ONEFLOW_GRAPH_SRCS} ${ONEFLOW_GRAPH_HDRS}
)

# TODO: graph doesn't need gflags
target_link_libraries(of_graph
    of_protoobj
    ${oneflow_exe_third_party_libs}
)
