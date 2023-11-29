# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.scaler_graph.IR import operator


class Parallelism:
    '''A abstract class for parallelism.
    parallelism will modify DNN graphs to implement parallel training.
    '''
    def __init__(self, devices):
        self.devices = devices

    def run_on_graph(self, graph):
        return True


class DataParallelism(Parallelism):
    '''duplicate the whole graph, insert allreduce between gradient and apply.
    '''
    def __init__(self, devices):
        super().__init__(devices)

    def run_on_graph(self, graph):
        # TODO(gbxu): add tags, delay graph manipulations
        for node in graph.nodes:
            if not isinstance(node.op, operator.ApplyOp):
                continue
            edge = node.in_edges[node.op.info["gradient_index"]]
            sc_op = operator.AllreduceOp()
            input_node_idxes = [(edge.src_node, edge.src_idx)]
            attrs = {
                "tensor_name": f"{edge.src_node.name}_allreduce",
                "T": edge.src_node.attrs["T"],
                "reduction": "sum",
                "num_devices": str(len(self.devices)),
            }
            node_name = f"{edge.src_node.name}_allreduce"
            graph.remove_edge(edge)
            node = graph.add_node_and_edge(node_name, sc_op, input_node_idxes,
                                           1, attrs)
            graph.add_edge(node, 0, edge.dest_node, edge.dest_idx)
        self.parallel_graphs = []
        self.parallel_graphs.extend(graph.copy() for _ in range(len(self.devices)))
        return True
