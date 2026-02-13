# -*- coding: utf-8 -*-
def classFactory(iface):
    from .qfat01_flow_estimator import QFAT01FlowEstimatorPlugin
    return QFAT01FlowEstimatorPlugin(iface)
