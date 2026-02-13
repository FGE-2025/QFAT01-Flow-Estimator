# -*- coding: utf-8 -*-
from qgis.PyQt import QtWidgets
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QCoreApplication
from pathlib import Path

from .qfat01_flow_estimator_dialog import QFAT01FlowEstimatorDialog

class QFAT01FlowEstimatorPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dlg = None

    def tr(self, message):
        return QCoreApplication.translate("QFAT01FlowEstimatorPlugin", message)

    def initGui(self):
        icon_path = str(Path(__file__).resolve().parent / "icon.png")
        self.action = QtWidgets.QAction(QIcon(icon_path), self.tr("QFAT01 Flow Estimator"), self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu(self.tr("&QFAT01 Flow Estimator"), self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        try:
            if self.dlg is not None:
                self.dlg._cleanup_canvas_items()
        except Exception:
            pass
        if self.action:
            self.iface.removePluginMenu(self.tr("&QFAT01 Flow Estimator"), self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action = None
        self.dlg = None

    def run(self):
        if self.dlg is None:
            self.dlg = QFAT01FlowEstimatorDialog(self.iface, self.iface.mainWindow())
        self.dlg.show()
        self.dlg.raise_()
        self.dlg.activateWindow()
