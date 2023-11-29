# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import google.protobuf.text_format
import os
import re
from pathlib import Path
from tensorflow.python import types_pb2, tensor_shape
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework.op_def_pb2 import OpDef
import tensorflow as tf
from tensorflow.python.util import compat
from tensorflow.python.framework.ops import Operation, Tensor
from tensorflow.python.framework.op_def_library import (
    _IsListValue,
    _MakeBool,
    _MakeFloat,
    _MakeInt,
    _MakeShape,
    _MakeStr,
    _MakeTensor,
    _MakeType,
)
from tensorflow.python.framework import kernels
from superscaler.scaler_graph.IR.graph import Graph
from superscaler.scaler_graph.IR.conversion.tensorflow_ops \
    import tf_op_map_to_sc_op, convert_to_tf_node
from superscaler.scaler_graph.IR.util import graph_util
from superscaler.scaler_graph.util.log import logger
__all__ = [
    "import_graph_from_tf_pbtxts", "get_tf_runtime_config",
    "export_graph_to_tf_file", "import_tensorflow_model", "set_dataset_paths"
]


def __set_device_info(graph_def):
    '''
    1. return CPU device info if no GPU kernel
    2. Heuristic A: prefer to place "generators" with their only
       consumers. If this is a node with no inputs and one output
       , we save this for a second pass, so that the consumer's
       placement is chosen.
    '''
    def support_GPU(tf_node):
        special_ops = ["MakeIterator", "IteratorV2", "IteratorGetNext"]
        if tf_node.op in special_ops:
            return False
        ignore_ops = ["NoOp"]
        if tf_node.op in ignore_ops:
            return True
        support_GPU = False
        kernel_list = kernels.get_registered_kernels_for_op(tf_node.op)
        if len(kernel_list.kernel) < 1:
            logger().error(f"no kernel for operator: {tf_node.op}")
            raise RuntimeError
        for kernel in kernel_list.kernel:
            is_suitable = True
            for constraint in kernel.constraint:
                if constraint.HasField("allowed_values"):
                    allowed_list = constraint.allowed_values.list.type
                    if tf_node.attr[constraint.name].type not in allowed_list:
                        is_suitable = False
                        break
            if is_suitable and kernel.device_type == "GPU":
                support_GPU = True
        return support_GPU

    no_input_nodes = []
    name_to_node = {}
    # rule 1
    for tf_node in graph_def.node:
        if not support_GPU(tf_node):
            tf_node.device = "/device:CPU:0"
        name_to_node[tf_node.name] = tf_node
        if len(tf_node.input) == 0:
            no_input_nodes.append(tf_node)
    # Heuristic A
    for tf_node in graph_def.node:
        for input in tf_node.input:
            name = input[1:] if input.startswith("^") else input.split(":")[0]
            if name_to_node[name] in no_input_nodes:
                name_to_node[name].device = tf_node.device


def __get_dtype_proto(node_def, op_def, output_arg):
    def with_number_attr(dtype):
        if len(output_arg.number_attr) != 0:
            for attr in op_def.attr:
                if attr.name == output_arg.number_attr:
                    return [dtype] * node_def.attr[attr.name].i
            raise AssertionError
        else:
            return dtype

    if len(output_arg.type_attr) != 0:
        for attr in op_def.attr:
            if attr.name == output_arg.type_attr:
                return with_number_attr(node_def.attr[attr.name].type)
        raise AssertionError
    elif len(output_arg.type_list_attr) != 0:
        for attr in op_def.attr:
            if attr.name == output_arg.type_list_attr:
                return list(node_def.attr[attr.name].list.type)
        raise AssertionError
    else:
        assert output_arg.type != types_pb2.DT_INVALID
        return with_number_attr(output_arg.type)


def __get_dtypes(tf_graph, node_def):
    '''parse tf dtypes.
    '''
    op_def = tf_graph._get_op_def(node_def.op)
    dtypes = [
        __get_dtype_proto(node_def, op_def, output_arg)
        for output_arg in op_def.output_arg
    ]
    if len(dtypes) == 1 and isinstance(dtypes[0], list):
        dtypes = dtypes[0]
    return [tf.as_dtype(dtype) for dtype in dtypes]


def __from_attr_proto(attr_value):
    '''parse tf node attributions.
    '''
    field_name = attr_value.WhichOneof("value")
    if field_name == "s":
        return attr_value.s
    elif field_name == "i":
        return attr_value.i
    elif field_name == "f":
        return attr_value.f
    elif field_name == "b":
        return attr_value.b
    elif field_name == "type":
        return tf.as_dtype(attr_value.type)
    elif field_name == "shape":
        return tensor_shape.as_shape(attr_value.shape)
    elif field_name == "tensor":
        return attr_value.tensor
    elif field_name == "func":
        return attr_value.func
    elif field_name == "placeholder":
        return attr_value.placeholder
    elif field_name == "list":
        list_value = attr_value.list
        if len(list_value.s) != 0:
            return list(list_value.s)
        elif len(list_value.i) != 0:
            return list(list_value.i)
        elif len(list_value.f) != 0:
            return list(list_value.f)
        elif len(list_value.b) != 0:
            return list(list_value.b)
        elif len(list_value.type) != 0:
            return [tf.as_dtype(value) for value in list_value.type]
        elif len(list_value.shape) != 0:
            return [tensor_shape.as_shape(value) for value in list_value.shape]
        elif len(list_value.tensor) != 0:
            return list(list_value.tensor)
        elif len(list_value.func) != 0:
            return list(list_value.func)
        else:
            return []


def import_graph_from_tf_pbtxts(file_paths, tf_runtime_config):
    '''convert tf pbtxts to sc graph.
    1. merge tf pbtxts into tf_graph_def;
    2. convert tf_graph_def to sc graph
    Return:
        SC graph
    '''
    tf_graph_def = tf.GraphDef()
    assert (len(file_paths) > 0)
    google.protobuf.text_format.Parse(
        Path(file_paths[0]).read_text(), tf_graph_def)
    for file_path in file_paths[1:]:
        google.protobuf.text_format.Merge(
            Path(file_path).read_text(), tf_graph_def)
    return __import_graph_from_tf_graph_def(tf_graph_def, tf_runtime_config)


def __import_graph_from_tf_graph_def(tf_graph_def, tf_runtime_config):
    def mark_runtime_info(sc_node, tf_runtime_config):
        if "sc_metadata" not in sc_node.attrs:
            sc_node.attrs["sc_metadata"] = {}
        if "runtime_config" not in sc_node.attrs["sc_metadata"]:
            sc_node.attrs["sc_metadata"]["runtime_config"] = {}
        node_runtime_config = sc_node.attrs["sc_metadata"]["runtime_config"]
        if sc_node.name in tf_runtime_config["inits"]:
            node_runtime_config["init"] = True
            tf_runtime_config["inits"].remove(sc_node.name)
        else:
            node_runtime_config["init"] = False
        if sc_node.name in tf_runtime_config["feeds"]:
            node_runtime_config["feed"] = True
            tf_runtime_config["feeds"].remove(sc_node.name)
        else:
            node_runtime_config["feed"] = False
        node_runtime_config["fetch"] = []
        for curr in tf_runtime_config["fetches"]:
            if sc_node.name == curr.split(":")[0]:
                if len(curr.split(":")) == 1:
                    node_runtime_config["fetch"].append(-1)
                else:
                    node_runtime_config["fetch"].append(int(
                        curr.split(":")[1]))
                tf_runtime_config["fetches"].remove(curr)
        if sc_node.name in tf_runtime_config["targets"]:
            node_runtime_config["target"] = True
            tf_runtime_config["targets"].remove(sc_node.name)
        else:
            node_runtime_config["target"] = False

    def add_sc_before_underscore(name):
        '''tf.import_graph_def() can't parse nodes with prefix "_",
        Add "sc" before "_".
        '''
        obj = re.match("^_.*$", name)
        if obj is not None:
            name = f"sc{name}"
        return name

    def add_sc_node(sc_graph, tf_node, name_to_node):
        if sc_graph.get_node_by_name(node.name) is not None:
            return
        input_node_idxes = []
        for input in tf_node.input:
            if input.startswith("^"):
                input_node_name = input[1:]
                # check control edge name
                input_node_name = add_sc_before_underscore(input_node_name)
                if sc_graph.get_node_by_name(input_node_name) is not None:
                    input_node = sc_graph.get_node_by_name(input_node_name)
                else:
                    add_sc_node(name_to_node[input_node_name])
                    input_node = sc_graph.get_node_by_name(input_node_name)
                index = -1
            else:
                names = input.split(":")
                assert len(names) in {1, 2}
                # check data edge name
                names[0] = add_sc_before_underscore(names[0])
                if sc_graph.get_node_by_name(names[0]) is not None:
                    input_node = sc_graph.get_node_by_name(names[0])
                else:
                    add_sc_node(name_to_node[names[0]])
                    input_node = sc_graph.get_node_by_name(names[0])
                index = 0 if len(names) == 1 else int(names[1])
            input_node_idxes.append((input_node, index))
        attrs = {
            attr_name: __from_attr_proto(tf_node.attr[attr_name])
            for attr_name in tf_node.attr
        }
        dtypes = __get_dtypes(tf_graph, tf_node)
        sc_node = sc_graph.add_node_and_edge(
            tf_node.name, tf_op_map_to_sc_op(tf_graph._get_op_def(tf_node.op)),
            input_node_idxes, len(dtypes), attrs)
        sc_node.attrs["tf"] = {}
        sc_node.attrs["tf"]["device"] = ""
        sc_node.attrs["tf"]["dtypes"] = dtypes
        mark_runtime_info(sc_node, tf_runtime_config)
        if tf_node.HasField("experimental_debug_info"):
            sc_node.attrs["tf"][
                "experimental_debug_info"] = node.experimental_debug_info

    sc_graph = Graph()
    tf_graph = tf.Graph()
    name_to_node = {}
    for node in tf_graph_def.node:
        node.name = add_sc_before_underscore(node.name)
        name_to_node[node.name] = node

    for tf_node in tf_graph_def.node:
        add_sc_node(sc_graph, tf_node, name_to_node)

    for key in tf_runtime_config.keys():
        for remaining in tf_runtime_config[key]:
            logger().error(f"{remaining} is not found in tf graph.")
        if len(tf_runtime_config[key]) > 0:
            raise RuntimeError

    for key in ["versions", "library"]:
        if tf_graph_def.HasField(key):
            sc_graph.attrs[key] = getattr(tf_graph_def, key)
    sc_graph.attrs["meta_graph"] = tf.MetaGraphDef()
    sc_graph.attrs["initialized_variables"] = {}
    sc_graph.attrs["lower_name_func"] = (lambda name: name.lower())
    return sc_graph


def get_tf_runtime_config(sc_graph):
    '''find some specific nodes for tf runtime.
    inits: nodes for backtracing all initialization nodes of variables.
    feeds: nodes without input, providing training data.
    fetches: feedback tensors for users, e.g. loss.
    targets: nodes without output, for backtracing all nodes needed to perform.
        e.g.: all applygradient ops, send op
    '''
    tf_runtime_config = {"inits": [], "feeds": [], "fetches": [], "targets": []}
    for sc_node in graph_util.get_output_nodes(sc_graph):
        node_runtime_config = sc_node.attrs["sc_metadata"]["runtime_config"]
        if node_runtime_config["init"]:
            tf_runtime_config["inits"].append(sc_node.name)
        if node_runtime_config["feed"]:
            tf_runtime_config["feeds"].append(sc_node.name)
        if node_runtime_config["target"]:
            tf_runtime_config["targets"].append(sc_node.name)
        for idx in node_runtime_config["fetch"]:
            if idx == -1:
                tf_runtime_config["fetches"].append(sc_node.name)
            else:
                tf_runtime_config["fetches"].append(sc_node.name + ":%d" %
                                                    (idx))
    return tf_runtime_config


def __sc_attrs_to_tf_attrs_proto(op_def, op_type_name, attrs):
    '''Convert attr values to AttrValue protos
    '''
    attr_protos = {}
    attr_defs = {attr_def.name: attr_def for attr_def in op_def.attr}
    for key, value in attrs.items():
        attr_value = tf.AttrValue()
        if key in attr_defs:
            attr_def = attr_defs[key]
        elif value is None:
            attr_protos[key] = attr_value
            continue
        else:
            attr_def = OpDef.AttrDef()
            if isinstance(value, (str, bytes)):
                attr_def.type = "string"
            elif isinstance(value, float):
                attr_def.type = "float"
            elif isinstance(value, bool):
                attr_def.type = "bool"
            # bool is a subclass of int, so we should check bool
            # before checking int
            elif isinstance(value, int):
                attr_def.type = "int"
            elif isinstance(value, tf.DType):
                attr_def.type = "type"
            elif isinstance(value, tf.TensorShape):
                attr_def.type = "shape"
            elif isinstance(value, tensor_pb2.TensorProto):
                attr_def.type = "tensor"
            elif isinstance(value, tf.NameAttrList):
                attr_def.type = "func"
            elif isinstance(value, list) and len(value) == 0:
                attr_value.list.SetInParent()
                attr_protos[key] = attr_value
                continue
            elif isinstance(value, list) and isinstance(
                    value[0], (str, bytes)):
                attr_def.type = "list(string)"
            elif isinstance(value, list) and isinstance(value[0], bool):
                attr_def.type = "list(bool)"
            # bool is a subclass of int, so we should check bool before
            # checking int
            elif isinstance(value, list) and isinstance(value[0], int):
                attr_def.type = "list(int)"
            elif isinstance(value, list) and isinstance(value[0], float):
                attr_def.type = "list(float)"
            elif isinstance(value, list) and isinstance(value[0], tf.DType):
                attr_def.type = "list(type)"
            elif isinstance(value, list) and isinstance(
                    value[0], tf.TensorShape):
                attr_def.type = "list(shape)"
            elif isinstance(value, list) and isinstance(
                    value[0], tensor_pb2.TensorProto):
                attr_def.type = "list(tensor)"
            else:
                logger().error(f"{value} has unsupported type")
                raise RuntimeError
        if attr_def.HasField("default_value") and value is None:
            attr_value.CopyFrom(attr_def.default_value)
            attr_protos[key] = attr_value
            continue
        if attr_def.type.startswith("list("):
            if not _IsListValue(value):
                logger().error(f"Expected list for attr {key}")
                raise TypeError
            if attr_def.has_minimum:
                if len(value) < attr_def.minimum:
                    logger().error(
                        "Attr '%s' of '%s' Op passed list of length %d "
                        "less than minimum %d." %
                        (key, op_type_name, len(value), attr_def.minimum))
                    raise ValueError
            attr_value.list.SetInParent()
        if attr_def.type == "string":
            attr_value.s = _MakeStr(value, key)
            if attr_def.HasField("allowed_values"):
                if attr_value.s not in attr_def.allowed_values.list.s:
                    logger().error(
                        "Attr '%s' of '%s' Op passed string '%s' not \
                            in: \"%s\"." % (
                            key,
                            op_type_name,
                            compat.as_text(attr_value.s),
                            '", "'.join(
                                map(compat.as_text,
                                    attr_def.allowed_values.list.s)),
                        ))
                    raise ValueError
        elif attr_def.type == "list(string)":
            attr_value.list.s.extend([_MakeStr(x, key) for x in value])
            if attr_def.HasField("allowed_values"):
                for x in attr_value.list.s:
                    if x not in attr_def.allowed_values.list.s:
                        logger().error(
                            "Attr '%s' of '%s' Op passed string '%s' not \
                                in: \"%s\"." % (
                                key,
                                op_type_name,
                                compat.as_text(x),
                                '", "'.join(
                                    map(compat.as_text,
                                        attr_def.allowed_values.list.s)),
                            ))
                        raise ValueError
        elif attr_def.type == "int":
            attr_value.i = _MakeInt(value, key)
            if attr_value.i < attr_def.minimum:
                if attr_def.has_minimum:
                    logger().error(
                        "Attr '%s' of '%s' Op passed %d less than minimum %d."
                        % (key, op_type_name, attr_value.i, attr_def.minimum))
                    raise ValueError
        elif attr_def.type == "list(int)":
            attr_value.list.i.extend([_MakeInt(x, key) for x in value])
        elif attr_def.type == "float":
            attr_value.f = _MakeFloat(value, key)
        elif attr_def.type == "list(float)":
            attr_value.list.f.extend([_MakeFloat(x, key) for x in value])
        elif attr_def.type == "bool":
            attr_value.b = _MakeBool(value, key)
        elif attr_def.type == "list(bool)":
            attr_value.list.b.extend([_MakeBool(x, key) for x in value])
        elif attr_def.type == "type":
            attr_value.type = _MakeType(value, attr_def)
        elif attr_def.type == "list(type)":
            attr_value.list.type.extend(
                [_MakeType(x, attr_def) for x in value])
        elif attr_def.type == "shape":
            attr_value.shape.CopyFrom(_MakeShape(value, key))
        elif attr_def.type == "list(shape)":
            attr_value.list.shape.extend([_MakeShape(x, key) for x in value])
        elif attr_def.type == "tensor":
            attr_value.tensor.CopyFrom(_MakeTensor(value, key))
        elif attr_def.type == "list(tensor)":
            attr_value.list.tensor.extend([_MakeTensor(x, key) for x in value])
        elif attr_def.type == "func":
            if isinstance(value, tf.NameAttrList):
                attr_value.func.CopyFrom(value)
            elif isinstance(value, compat.bytes_or_text_types):
                attr_value.func.name = value
            else:
                value.add_to_graph(tf.get_default_graph())
                attr_value.func.name = value.name
        else:
            logger().error(f"Unrecognized Attr type {attr_def.type}")
            raise TypeError

        attr_protos[key] = attr_value
    return attr_protos


def export_graph_to_tf_file(sc_graph, file_path=None):
    '''convert sc graph to tf graph
    TODO(gbxu): the library file path should be configurable.
    '''
    proj_path = os.environ["SUPERSCLAR_PATH"]
    lib_path = f"{proj_path}/lib/libsuperscaler_pywrap.so"
    if os.path.exists(lib_path):
        tf.load_library(lib_path)
    else:
        logger().error(f"The library file {lib_path} does not exist.")
        raise RuntimeError
    tf_graph = tf.Graph()
    graph_def = tf_graph.as_graph_def(add_shapes=True)
    for key in ["versions", "library"]:
        if key in sc_graph.attrs:
            getattr(graph_def, key).CopyFrom(sc_graph.attrs[key])
    for sc_node in sc_graph.nodes:
        convert_to_tf_node(sc_node)
        attrs = {
            name: value
            for name, value in sc_node.attrs.items()
            if name not in ["tf", "sc_metadata"]
        }
        tf_node = graph_def.node.add()
        tf_node.name = sc_node.name
        tf_node.op = sc_node.op.original_name
        tf_node.device = sc_node.attrs["tf"]["device"]
        if "experimental_debug_info" in sc_node.attrs["tf"]:
            tf_node.experimental_debug_info.CopyFrom(
                sc_node.attrs["experimental_debug_info"])
        for name, attr_value in __sc_attrs_to_tf_attrs_proto(
                tf_graph._get_op_def(tf_node.op), tf_node.op, attrs).items():
            tf_node.attr[name].CopyFrom(attr_value)

        for in_edge in sc_node.in_edges:
            if in_edge.src_idx == -1:
                in_edge_str = f"^{in_edge.src_node.name}"
            elif in_edge.src_idx == 0:
                in_edge_str = f"{in_edge.src_node.name}"
            else:
                in_edge_str = f"{in_edge.src_node.name}:{in_edge.src_idx}"
            tf_node.input.append(in_edge_str)
    # add devices info
    __set_device_info(graph_def)
    # add shapes
    output_graph = tf.Graph()
    with output_graph.as_default():
        tf.import_graph_def(graph_def, name="")
    graph_def = output_graph.as_graph_def(add_shapes=True)
    # dump graph as pbtxt
    graph_pbtxt = google.protobuf.text_format.MessageToString(graph_def)
    if file_path is not None:
        file = Path(file_path)
        file.write_text(graph_pbtxt)
    return graph_pbtxt


def import_tensorflow_model(session_run_params,
                            graph=None,
                            reset_default_graph=True):
    '''import tensorflow graph according to init_params, run_params
        1. run tensorflow model via init_params, run_params in session;
        2. dump graph from tensorflow model.
    Args:
        session_run_params: dict.
            The key "init_params" maps to a list of ops to initilizaton.
            The key "run_params" maps to a list of ops to training.
            example: session_run_params = {
                "init_params" = [iterator.initializer,
                    tf.global_variables_initializer()],
                "run_params" = [optimizer.minimize(loss_op), loss_op]
            }
        graph: the model graph. If None, tf.get_default_graph() will be called.
        reset_default_graph: A flag to clean the default graph each time.
            It's set True by default.
    Returns:
        A SC Graph.
    '''
    tf_runtime_config = {"inits": [], "feeds": [], "fetches": [], "targets": []}
    for init_op in session_run_params["init_params"]:
        if isinstance(init_op, (Operation, Tensor)):
            tf_runtime_config["inits"].append(
                init_op.name.split(":")[0])  # op_name only
        else:
            raise RuntimeError
    for run_op in session_run_params["run_params"]:
        if isinstance(run_op, Operation):
            tf_runtime_config["targets"].append(run_op.name)  # op_name
        elif isinstance(run_op, Tensor):
            tf_runtime_config["fetches"].append(run_op.name)  # op_name:index
        else:
            raise RuntimeError

    if graph is None:
        graph = tf.get_default_graph()
    graph_def = graph.as_graph_def(add_shapes=True)
    pruned_graph_def = tf.compat.v1.graph_util.extract_sub_graph(
        graph_def, [
            op.name.split(":")[0] for op in session_run_params["init_params"] +
            session_run_params["run_params"]
        ])
    if reset_default_graph:
        tf.reset_default_graph()
    return __import_graph_from_tf_graph_def(
        pruned_graph_def, tf_runtime_config
    )


def set_dataset_paths(graph, paths):
    '''
    1. convert tf_graph string to sc graph.
    2. set dataset paths.
    Return:
        is_successed
    '''
    tf_graph_def = tf.GraphDef()
    google.protobuf.text_format.Parse(graph, tf_graph_def)
    count = 0
    for tf_node in tf_graph_def.node:
        if tf_node.op == "Const" and "value" in tf_node.attr:
            path_flag = tf_node.attr["value"].tensor.string_val
            if len(path_flag) != 1:
                continue
            path_flag = path_flag[0]
            if len(path_flag.split(b':')) > 1 and path_flag.split(
                    b':')[0] == b'DATASET_PATH':
                idx = int(path_flag.split(b':')[1])
                tf_node.attr["value"].tensor.string_val[0] = _MakeStr(
                    paths[idx], 'string_val')
                count += 1
    assert (count == len(paths))
    return google.protobuf.text_format.MessageToString(tf_graph_def)
