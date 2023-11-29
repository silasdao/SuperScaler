# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.scaler_graph.util.log import logger


class Operation:
    pass


class Parallelizer:
    '''parallelizer: a tool to optimize and apply the parallelisms.
    #TODO(gbxu): re-design the workflow of Parallelizer.
    '''
    def __init__(self, sc_graph):
        self.parallelisms = []
        self.graphs = [
            sc_graph,
        ]
        self.support_operations = {}

    def register_parallelism(self, parallelism):
        '''register parallelism into parallelizer
        '''
        self.parallelisms.append(parallelism)

    def run_parallelisms(self):
        for parallelism in self.parallelisms:
            for graph in self.graphs:
                if not parallelism.run_on_graph(graph):
                    logger().error(
                        f"failed when {parallelism.__class__.__name__} runs on a graph."
                    )
                    raise RuntimeError
            self.graphs = parallelism.parallel_graphs
            logger().info(f"Run {parallelism.__class__.__name__} : successed.")

        self.finalize()

    def finalize(self):
        '''optimize tags and apply graph manipulations.
        '''
        pass

    def register_operation(self, customized_operation):
        pass
