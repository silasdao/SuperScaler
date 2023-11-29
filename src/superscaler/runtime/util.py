#! /usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import subprocess
from collections import defaultdict


def run_shell_cmd(cmd):
    subprocess.check_output(cmd, shell=True)


def distribute_resources(deployment_setting,
                         local_resource_dir='/tmp/superscaler',
                         remote_resource_dir='/tmp/superscaler'):
    """
    this helper function helps to distribute resources to remote workers.

    @deployment_setting: a dict for specify how many workers over each host ip
    @local_resource_dir: generated resources by the master, it's organized in\
        the form of many global rank indexed folders in which contains all the\
        necessary per-process running resources
    @remote_working_dir: this is where the specified resource will be\
        delivered to on the remote host, it should be specified\
        in a full-path manner
    e.g., deployment_setting = {
        '10.0.0.21': 2,
    }
    """
    if not os.path.exists(local_resource_dir):
        raise Exception(f'local_resource_dir: {local_resource_dir} is not existed!')

    for ip in deployment_setting.keys():
        if ip == "localhost":
            run_shell_cmd(f'rsync -az {local_resource_dir} {remote_resource_dir}')
        else:
            # Remote sync runtime files
            run_shell_cmd(f'rsync -az {local_resource_dir} {ip}:{remote_resource_dir}')

    return os.path.join(remote_resource_dir,
                        os.path.basename(local_resource_dir))


def launch(rank2ip, rank2cmd):
    """
    this helper function helps to launch cmds in a MPMD manner,\
        and it's assumed all the ssh passwdless connections are\
        built between workers before its use
    @rank2ip: a list of process ip, indexed by its global rank,\
        which means rank 0 will be assigned to the corresponding ip
    @rank2cmd:  a list of per process to-be-executed cmd
    """
    def parse_host_args(rank2ip):
        hosts_and_slots = defaultdict(int)
        for x in rank2ip:
            hosts_and_slots[x] += 1
        return hosts_and_slots

    hosts_and_slots = parse_host_args(rank2ip)

    mpirun_command = (
        'mpirun --allow-run-as-root --tag-output '
        '{cmds} '.format(
            cmds=' : '.join(
                '-np 1 -host {ip_slots} {cmd}'.format(
                    ip_slots=f'{ip}:{str(hosts_and_slots[ip])}', cmd=(cmd)
                )
                for ip, cmd in zip(rank2ip, rank2cmd)
            )
        )
    )

    run_shell_cmd(mpirun_command)
