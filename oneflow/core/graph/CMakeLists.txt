file(GLOB_RECURSE ONEFLOW_GRAPH_HDRS "*.h")
file(GLOB_RECURSE ONEFLOW_GRAPH_SRCS "*.cpp")
add_library(of_graph
    ${ONEFLOW_GRAPH_HDRS} ${ONEFLOW_GRAPH_SRCS}
)
add_dependencies(of_graph of_cfgobj)
add_dependencies(of_graph of_protoobj)
