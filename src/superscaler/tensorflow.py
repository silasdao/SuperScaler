# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.superscaler import Superscaler, SuperscalerError
from superscaler.scaler_graph import tf_adapter, Parallelizer, DataParallelism
from superscaler.plan_gen import TFParser, PlanGenerator, SuperScalerAdapter
from superscaler.runtime.util import distribute_resources, launch
import tensorflow as tf
import argparse
from functools import wraps


@wraps(tf.data.TFRecordDataset)
def TFRecordDataset(*args, **kwargs):
    if "filenames" in kwargs:
        for i, _ in enumerate(kwargs["filenames"]):
            kwargs["filenames"][i] = f"DATASET_PATH:{str(i)}"
    else:
        assert args
        args_copy = list(args)
        args_copy[0] = "DATASET_PATH:0"
    return tf.data.TFRecordDataset(*args_copy, **kwargs)


class tensorflow(Superscaler):
    """ Wrapper class for the Superscaler API for tensorflow framework. """

    def __init__(self):
        super().__init__()
        self._plan_parser = TFParser()

    def _init_partition_graphs(self, session_run_params, strategy):
        """ A function that partition tensorflow graph by parallelism strategy.

        Args:
          session_run_params: a dict contains "init_params" and "run_params"
            operator of tensorflow graph
          strategy: distributed training strategy including data parallelism,
            model parallelism and pipeline parallelism.
        """
        if not isinstance(
                session_run_params,
                dict) or "init_params" not in session_run_params.keys(
                ) or "run_params" not in session_run_params.keys():
            raise SuperscalerError('session_run_params must be a dict with \
                    keys "init_params" and "run_params".')
        for node in session_run_params["init_params"]:
            if not isinstance(node, tf.Operation):
                raise SuperscalerError(
                    'nodes in session_run_params["init_params"] \
                        must be tf.Operation')
        for node in session_run_params["run_params"]:
            if not isinstance(node, tf.Operation) and not isinstance(
                    node, tf.Tensor):
                raise SuperscalerError(
                    'nodes in session_run_params["run_params"] must be \
                    tf.Operation or tf.Tensor')
        if not isinstance(strategy, DataParallelism):
            raise SuperscalerError("Unsupport parallelism strategy")

        # run_parallelisms
        merged_sc_graph = tf_adapter.import_tensorflow_model(
            session_run_params)
        parallelizer = Parallelizer(merged_sc_graph)
        parallelizer.register_parallelism(strategy)
        parallelizer.run_parallelisms()

        # Convert partition_graphs into tf_protobuf
        self._partition_graphs = []
        self._partition_graphs.extend(
            tf_adapter.export_graph_to_tf_file(graph)
            for graph in parallelizer.graphs
        )
        self._graph_count = len(parallelizer.graphs)
        self._graph_config = tf_adapter.get_tf_runtime_config(merged_sc_graph)

    def _init_communication_plan(self, resource_pool, communication_DSL):
        """ A function that generate communication_plan from resource_pool.

        Args:
          resource_pool: JSON file specifying hardware description and network
            topology.
          communication_DSL: domain specific language to decribe communication
        """
        if not isinstance(resource_pool, str):
            raise SuperscalerError("resource_pool should be inited from file")
        if not isinstance(communication_DSL, str):
            raise SuperscalerError("communication_DSL should be str")

        # init plan_generator
        self._resoure_pool.init_from_yaml(resource_pool)
        devices = [f"device_{str(i)}" for i in range(self._graph_count)]
        nodelist = self._plan_parser.parse_graphs(
            self._partition_graphs, devices, load_from_memory=True)

        # Generate communication_plan
        plan_generator = PlanGenerator(nodelist, self._resoure_pool)
        plan = plan_generator.get_execution_plan('Allreduce',
                                                 communication_DSL)

        # Adapt plan for Superscaler
        self._plan_adapter = SuperScalerAdapter()
        self._plan_adapter.set_plan(plan)
        self._communication_plan = self._plan_adapter.adapt_plan()

    def _set_dataset_paths(self, dataset_paths):
        """
        update dataset path in graph
        """
        for i, (graph, plan) in enumerate(
                zip(self._partition_graphs, self._assigned_plan)):
            if plan['ip'] in dataset_paths.keys():
                paths = dataset_paths[plan['ip']].pop()
                new_graph = tf_adapter.set_dataset_paths(graph, paths)
                self._partition_graphs[i] = new_graph

    def run(self, args):
        """ A function that performs distributed training.
            This function is avaliable when self.is_initialized() is True
        """

        if self.is_initialized() is not True:
            raise SuperscalerError("Superscaler must be run \
                                    after initialization is complete")
        if not isinstance(args, argparse.Namespace) or\
               not isinstance(args.steps, int) or\
               not isinstance(args.interval, int) or\
               not isinstance(args.print_info, bool) or\
               not isinstance(args.print_fetches_targets, bool):
            raise SuperscalerError("Superscaler runtime argument illegal")

        deployment_config, rank2ip =\
                self._plan_assigner.get_deployment_config(self._assigned_plan)
        remote_resource_dir = distribute_resources(deployment_config,
                                                   self._working_dir)
        print(remote_resource_dir)
        cmd_per_worker = [
            'python -m superscaler.runtime.tensorflow.runner '
            '--model_dir_prefix {resource_dir}/{grank} '
            '--steps {steps} '
            '--interval {interval} '
            '--print_info {print_info} '
            '--print_fetches_targets {print_fetches_targets} '
            .format(resource_dir=remote_resource_dir,
                    grank=grank,
                    steps=args.steps,
                    interval=args.interval,
                    print_info=args.print_info,
                    print_fetches_targets=args.print_fetches_targets)
            for grank, _ in enumerate(rank2ip)
        ]
        launch(rank2ip, cmd_per_worker)
