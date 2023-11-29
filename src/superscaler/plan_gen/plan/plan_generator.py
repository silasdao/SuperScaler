# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''
Author: v-cluo
Version: 0.1
Update date: 07/22/2019

Main model of Plan Generator project.
This module will parsed nodelist, then generate the execution plan

v0.1 supported function:
Introduce import package.
'''

from superscaler.plan_gen.plan.plan_mapper import GPURoundRobinMapper
from superscaler.plan_gen.plan.plan_pool import PlanPool
from superscaler.plan_gen.plan.ring_allreduce_plan import RingAllreducePlan
from superscaler.plan_gen.plan.reduce_broadcast_allreduce_plan import \
     ReduceBroadcastAllreducePlan
from superscaler.plan_gen.plan.plan_manager import PlanManager


class PlanGenerator():
    def __init__(self, nodelist, resource_pool):
        ''' Init PlanGenerator with nodelist and a resource_pool

        Args:
            nodelist: a list node parsed from machine learning platform
            resource_pool: a ResourcePool class containing device info and
                router info
        '''

        # Init resource_pool
        self.__resource_pool = resource_pool
        # Init nodelist
        self.__nodelist = nodelist
        # Init PlanPool
        self.__plan_pool = PlanPool()
        self.__plan_pool.add_plan(RingAllreducePlan(plan_name='ring'))
        self.__plan_pool.add_plan(
            ReduceBroadcastAllreducePlan(plan_name='ReduceBroadcast'))
        # Init plan mapper
        # TODO Introduce mapping plan here
        self.__mapper = GPURoundRobinMapper(resource_pool)
        # Init PlanManager by PlanPool and PlanMapper
        self.__planmanager = PlanManager(self.__plan_pool, self.__mapper)

    def get_execution_plan(self, plan_type, plan_name):
        ''' Generate the execution plan by plan_type and plan_name
        Args:
            plan_type: string, e.g. Default
            plan_name: string, e.g. Default_Plan
        '''
        return self.__planmanager.get_execution_plan(
            self.__nodelist, plan_type, plan_name
        )

    def get_links_info(self):
        return self.__resource_pool.get_links_as_list()

    def get_routing_info(self):
        return self.__mapper.route_info

    def get_device_info(self):
        return self.__resource_pool.get_computational_hardware_as_list()
