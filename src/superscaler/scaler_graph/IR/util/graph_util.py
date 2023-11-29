# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
from superscaler.scaler_graph.IR.node import CompositeNode
from superscaler.scaler_graph.util.log import logger


def get_output_nodes(graph):
    '''get the nodes which have no output.
    They can be "fetch node".
    '''
    upstream_nodes = set(
        itertools.chain.from_iterable(
            map(
                lambda node: get_upstream_nodes(node),
                graph.nodes,
            )))
    return set(graph.nodes) - upstream_nodes


def reverse_DFS(graph):
    '''get the orderd nodes via topological sort.
    '''
    output_nodes = get_output_nodes(graph)
    temp_nodes = set()
    ordered_nodes = []

    def visit(current_node):
        if current_node in ordered_nodes:
            return
        elif current_node in temp_nodes:
            logger().error(f"there is a cycle in graph: {current_node.name}")
            raise RuntimeError
        else:
            temp_nodes.add(current_node)
            for input_node in get_upstream_nodes(current_node):
                visit(input_node)
            temp_nodes.remove(current_node)
            ordered_nodes.append(current_node)

    for current_node in output_nodes:
        visit(current_node)

    return ordered_nodes


def get_upstream_nodes(node):
    '''get all input nodes of current node.
    '''
    if isinstance(node, CompositeNode):
        logger().error("We can't support CompositeNode now.")
        raise RuntimeError
    return {edge.src_node for edge in node.in_edges}
