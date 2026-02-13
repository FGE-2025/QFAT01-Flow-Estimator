# -*- coding: utf-8 -*-
from pathlib import Path
import math
import re
import csv
from datetime import datetime

from qgis.PyQt import QtWidgets, uic
from qgis.PyQt.QtCore import Qt, QObject, QEvent
from qgis.PyQt.QtGui import QCursor, QColor
from qgis.PyQt.QtWidgets import QMessageBox, QFileDialog, QMenu

from qgis.core import (
    QgsProject, QgsWkbTypes, QgsFeature, QgsField,
    QgsVectorLayer, QgsGeometry, QgsPointXY, QgsCoordinateTransform
)
from qgis.gui import QgsMapTool, QgsRubberBand

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


class _XSLineCaptureTool(QgsMapTool):
    def __init__(self, canvas, on_finished, on_cancelled):
        super().__init__(canvas)
        self.canvas = canvas
        self.on_finished = on_finished
        self.on_cancelled = on_cancelled
        self.points = []
        self.rb = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
        self.rb.setColor(QColor(200, 0, 0, 255))  # red
        self.rb.setWidth(1)

    def activate(self):
        super().activate()
        self.canvas.setCursor(QCursor(Qt.CrossCursor))

    def deactivate(self):
        try:
            self.rb.reset(QgsWkbTypes.LineGeometry)
        except Exception:
            pass
        super().deactivate()

    def canvasPressEvent(self, e):
        if e.button() == Qt.RightButton:
            self._finish()
            return
        p = self.toMapCoordinates(e.pos())
        self.points.append(QgsPointXY(p))
        self.rb.addPoint(QgsPointXY(p), True)

    def canvasMoveEvent(self, e):
        if not self.points:
            return
        p = self.toMapCoordinates(e.pos())
        try:
            self.rb.movePoint(QgsPointXY(p))
        except Exception:
            pass

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self._cancel()
            return
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._finish()
            return

    def _cancel(self):
        self.points = []
        try:
            self.rb.reset(QgsWkbTypes.LineGeometry)
        except Exception:
            pass
        if self.on_cancelled:
            self.on_cancelled()

    def _finish(self):
        if len(self.points) < 2:
            self._cancel()
            return
        geom = QgsGeometry.fromPolylineXY(self.points)
        if self.on_finished:
            self.on_finished(geom)


class _LabelDblClickFilter(QObject):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
            try:
                self.handler(obj.objectName())
            except Exception:
                pass
            return True
        return False


class QFAT01FlowEstimatorDialog(QtWidgets.QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        ui_path = Path(__file__).resolve().parent / "flow_estimator.ui"
        uic.loadUi(str(ui_path), self)

        # State
        self.dem_layer = None
        self.xs_geom = None  # QgsGeometry in project CRS
        self.samples_full = []  # list of dict {x, z}
        self.bed_datum = None

        # persistent XS rubberband (stays until clear/new/close)
        self._rb_xs = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.LineGeometry)
        self._rb_xs.setColor(QColor(200, 0, 0, 255))  # red
        self._rb_xs.setWidth(1)

        # Plot state / toggles
        self._mpl_canvas = None
        self._mpl_ax = None
        self._mpl_toolbar = None
        self._artists = {}
        self._plot_quality = "adaptive"
        self._last_plot_args = {}
        self._plot_flags = {
            "show_bed": True,
            "show_wsl": True,
            "show_crest": True,
            "show_p": True,
            "show_a": True,
            "show_excl": True,
            "show_points": False,
        }
        self._last_motion_t = 0.0
        self._motion_min_dt = 1.0/25.0  # ~25 Hz

        # Exclusion ranges cache
        self._excl_ranges = []  # list of (a,b)
        self._allowed_ranges = []  # complement within [0, L]

        # Init plot
        self._init_plot()

        # Wire signals
        self.btnDemUseActive.clicked.connect(self.on_use_active_dem)
        self.btnDemClear.clicked.connect(self.on_clear_dem)

        self.btnDrawTempXS.clicked.connect(self.on_draw_xs)
        self.btnUseSelectedXS.clicked.connect(self.on_use_selected_xs)
        self.btnClearXS.clicked.connect(self.on_clear_xs)
        self.btnSaveXS.clicked.connect(self.on_save_xs)

        self.cboSampleSpacing.currentIndexChanged.connect(lambda *_: self.on_sampling_params_changed())
        self.edtSampleSpacingCustom.textChanged.connect(lambda *_: self.on_sampling_params_changed())
        self.cboChainage.currentIndexChanged.connect(lambda *_: self.on_chainage_changed())

        self.edtQ.textChanged.connect(lambda *_: self.on_params_changed())
        self.edtDepth.textChanged.connect(lambda *_: self.on_params_changed())
        self.edtSlopeUser.textChanged.connect(lambda *_: self.on_params_changed())
        self.edtN.textChanged.connect(lambda *_: self.on_params_changed())

        # exclusion input
        self.edtExcludeRanges.setPlaceholderText("Example: 0,10;70,999")
        self.edtExcludeRanges.textChanged.connect(lambda *_: self.on_params_changed())

        self.rdoInputQOutputDepth.toggled.connect(lambda *_: self.on_mode_changed())
        self.rdoInputDepthOutputQ.toggled.connect(lambda *_: self.on_mode_changed())

        self.btnCreateWettedPoly.clicked.connect(self.on_create_wetted_polygon)
        self.btnExportCSV.clicked.connect(self.on_export_csv)
        self.btnCopySummary.clicked.connect(self.on_copy_summary)

        # defaults (in UI already, but enforce safely)
        if not self.edtSlopeUser.text().strip():
            self.edtSlopeUser.setText("0.01")
        if not self.edtN.text().strip():
            self.edtN.setText("0.03")
        if not self.edtQ.text().strip():
            self.edtQ.setText("10")
        if not self.edtDepth.text().strip():
            self.edtDepth.setText("0.5")

        # mode enable
        self.on_mode_changed()

        # Custom sample spacing enable logic
        self._update_custom_spacing_enabled()

        # Double-click toggles on result labels
        self._install_label_toggles()

        # Default DEM from active raster
        self._try_set_default_dem_from_active()

        self._set_status("Status: No XS")
        self._clear_results()

    # ------------- lifecycle / cleanup -------------
    def closeEvent(self, event):
        try:
            self._cleanup_canvas_items()
        except Exception:
            pass
        super().closeEvent(event)

    def _cleanup_canvas_items(self):
        try:
            self._rb_xs.reset(QgsWkbTypes.LineGeometry)
        except Exception:
            pass

    # ------------- UI helpers -------------
    def _set_status(self, text):
        if hasattr(self, "lblStatus"):
            self.lblStatus.setText(text)

    def _set_notes(self, text):
        if hasattr(self, "lblResultNotes"):
            self.lblResultNotes.setText(f"Notes: {text}")

    def _clear_results(self):
        for name in ("edtQOut","edtDepthOut","edtWSL","edtA","edtP","edtR","edtV","edtFreeboard","edtCrestLow"):
            w = getattr(self, name, None)
            if w:
                w.setText("--")
        if hasattr(self, "lblPOverlay"):
            self.lblPOverlay.setText("P = -- m")
        self._set_notes("--")

    def _parse_float(self, txt):
        try:
            return float(str(txt).strip())
        except Exception:
            return None

    def on_mode_changed(self):
        # Q->WSL uses Q input; Depth->Q uses Depth input
        depth_mode = self.rdoInputDepthOutputQ.isChecked()
        self.edtDepth.setEnabled(depth_mode)
        self.edtQ.setEnabled(not depth_mode)
        self.on_params_changed()

    # ------------- DEM selection -------------
    def _try_set_default_dem_from_active(self):
        active = self.iface.activeLayer()
        if active and active.type() == active.RasterLayer:
            self.dem_layer = active
            self.edtDem.setText(active.name())

    def on_use_active_dem(self):
        active = self.iface.activeLayer()
        if not active or active.type() != active.RasterLayer:
            QMessageBox.information(self, "QEP18", "Active layer is not a raster layer.")
            return
        self.dem_layer = active
        self.edtDem.setText(active.name())
        self._set_status("Status: DEM set from active layer.")
        if self.xs_geom is not None:
            self._sample_xs()
            self.on_params_changed()

    def on_clear_dem(self):
        self.dem_layer = None
        self.edtDem.setText("")
        self.samples_full = []
        self.bed_datum = None
        self._set_status("Status: DEM cleared.")
        self._clear_results()
        self._plot_profile()

    # ------------- XS actions -------------
    def on_draw_xs(self):
        canvas = self.iface.mapCanvas()
        self._set_status("Status: Draw XS (left click points, right click/Enter to finish, ESC to cancel).")

        def finished(geom):
            self.iface.mapCanvas().unsetMapTool(tool)
            self._set_xs_geometry(geom, from_draw=True)

        def cancelled():
            self.iface.mapCanvas().unsetMapTool(tool)
            self._set_status("Status: Draw cancelled.")

        tool = _XSLineCaptureTool(canvas, finished, cancelled)
        canvas.setMapTool(tool)

    def _set_xs_geometry(self, geom, from_draw=False):
        if geom is None or geom.isEmpty():
            return
        self.xs_geom = geom

        # Persist temp XS display (red)
        try:
            self._rb_xs.reset(QgsWkbTypes.LineGeometry)
            pl = geom.asPolyline()
            if not pl:
                mpl = geom.asMultiPolyline()
                if mpl and mpl[0]:
                    pl = mpl[0]
            for i, pt in enumerate(pl):
                self._rb_xs.addPoint(QgsPointXY(pt), i == (len(pl)-1))
        except Exception:
            pass

        self._set_status(f"Status: XS {'drawn' if from_draw else 'loaded'} ({geom.length():.1f} m).")

        if self.dem_layer is None:
            self._clear_results()
            self._set_notes("Select a DEM to sample profile.")
            self._plot_profile()
            return

        self._sample_xs()
        self.on_params_changed()

    def on_use_selected_xs(self):
        layer = self.iface.activeLayer()
        if not layer or layer.type() != layer.VectorLayer:
            QMessageBox.information(self, "QEP18", "Active layer is not a vector layer.")
            return
        if layer.geometryType() != QgsWkbTypes.LineGeometry:
            QMessageBox.information(self, "QEP18", "Active layer is not a line layer.")
            return
        sel = layer.selectedFeatures()
        if len(sel) != 1:
            QMessageBox.information(self, "QEP18", "Please select exactly one XS feature.")
            return
        geom = sel[0].geometry()
        if not geom or geom.isEmpty():
            QMessageBox.information(self, "QEP18", "Selected feature has no valid geometry.")
            return

        # Transform to project CRS if needed
        try:
            prj_crs = QgsProject.instance().crs()
            lyr_crs = layer.crs()
            if lyr_crs.isValid() and prj_crs.isValid() and lyr_crs.authid() != prj_crs.authid():
                ct = QgsCoordinateTransform(lyr_crs, prj_crs, QgsProject.instance())
                geom = QgsGeometry(geom)
                geom.transform(ct)
        except Exception:
            pass

        self._set_xs_geometry(geom, from_draw=False)

    def on_clear_xs(self):
        self.xs_geom = None
        self.samples_full = []
        self.bed_datum = None
        self._clear_results()
        self._set_status("Status: No XS")
        self._plot_profile()
        # clear temp line on canvas too
        try:
            self._rb_xs.reset(QgsWkbTypes.LineGeometry)
        except Exception:
            pass

    def on_save_xs(self):
        if self.xs_geom is None:
            QMessageBox.information(self, "QEP18", "No XS to save.")
            return

        prj = QgsProject.instance()
        prj_crs = prj.crs()
        crs_auth = prj_crs.authid() if prj_crs and prj_crs.isValid() else "EPSG:3857"

        # Reuse or create memory layer
        layer = None
        for lyr in prj.mapLayers().values():
            if lyr.name() == "QEP18_XS" and lyr.type() == lyr.VectorLayer and lyr.geometryType() == QgsWkbTypes.LineGeometry:
                layer = lyr
                break

        if layer is None:
            layer = QgsVectorLayer(f"LineString?crs={crs_auth}", "QEP18_XS", "memory")
            prov = layer.dataProvider()
            prov.addAttributes([
                QgsField("xs_name", 50),
                QgsField("dem_name", 80),
                QgsField("spacing", 30),
                QgsField("S", 20),
                QgsField("n", 20),
                QgsField("exclude", 120),
                QgsField("created", 30),
            ])
            layer.updateFields()
            prj.addMapLayer(layer)

        prov = layer.dataProvider()
        feat = QgsFeature(layer.fields())
        feat.setGeometry(QgsGeometry(self.xs_geom))

        feat["xs_name"] = f"XS_{layer.featureCount()+1:03d}"
        feat["dem_name"] = self.dem_layer.name() if self.dem_layer else ""
        feat["spacing"] = self._spacing_text_for_save()
        feat["S"] = self.edtSlopeUser.text()
        feat["n"] = self.edtN.text()
        feat["exclude"] = (self.edtExcludeRanges.text() or "").strip()
        feat["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ok, _ = prov.addFeatures([feat])
        layer.updateExtents()

        if ok:
            QMessageBox.information(self, "QEP18", "XS saved to layer 'QEP18_XS'.")
        else:
            QMessageBox.warning(self, "QEP18", "Failed to save XS. (Provider addFeatures failed)")

    # ------------- sampling / spacing / chainage -------------
    def _update_custom_spacing_enabled(self):
        txt = self.cboSampleSpacing.currentText()
        is_custom = "Custom" in txt
        self.edtSampleSpacingCustom.setEnabled(is_custom)

    def on_sampling_params_changed(self):
        self._update_custom_spacing_enabled()
        if self.xs_geom is None or self.dem_layer is None:
            return
        self._sample_xs()
        self.on_params_changed()

    def on_chainage_changed(self):
        if self.xs_geom is None or self.dem_layer is None:
            return
        if not self.samples_full:
            self._mpl_canvas.draw_idle()
            self._reset_plot_tools()
            return
    def _spacing_text_for_save(self):
        t = self.cboSampleSpacing.currentText()
        if "Custom" in t:
            return f"Custom {self.edtSampleSpacingCustom.text()} m"
        return t

    def _current_step_m(self):
        if self.dem_layer is None:
            return None
        t = self.cboSampleSpacing.currentText()
        px = (abs(self.dem_layer.rasterUnitsPerPixelX()) + abs(self.dem_layer.rasterUnitsPerPixelY())) / 2.0
        if not px or px <= 0:
            px = 1.0

        if "Auto" in t:
            return px
        if "0.5" in t:
            return 0.5 * px
        if "1 x" in t:
            return 1.0 * px
        if "2 x" in t:
            return 2.0 * px
        if "5 m" in t:
            return 5.0
        if "10 m" in t:
            return 10.0
        if "Custom" in t:
            v = self._parse_float(self.edtSampleSpacingCustom.text())
            return v if v and v > 0 else px
        return px

    def _parse_exclusions(self, length_m):
        s = (self.edtExcludeRanges.text() or "").strip()
        if not s:
            return []
        s = re.sub(r"\s+", "", s)
        parts = [p for p in s.split(";") if p]
        ranges = []
        for p in parts:
            if "," not in p:
                continue
            a, b = p.split(",", 1)
            try:
                fa = float(a)
                fb = float(b)
            except Exception:
                continue
            lo, hi = (fa, fb) if fa <= fb else (fb, fa)
            lo = max(0.0, min(length_m, lo))
            hi = max(0.0, min(length_m, hi))
            if hi <= lo:
                continue
            ranges.append((lo, hi))
        if not ranges:
            return []
        ranges.sort()
        merged = [ranges[0]]
        for lo, hi in ranges[1:]:
            plo, phi = merged[-1]
            if lo <= phi + 1e-9:
                merged[-1] = (plo, max(phi, hi))
            else:
                merged.append((lo, hi))
        return merged

    def _compute_allowed_ranges(self, length_m, excl):
        if not excl:
            return [(0.0, length_m)]
        allowed = []
        cur = 0.0
        for lo, hi in excl:
            if lo > cur:
                allowed.append((cur, lo))
            cur = max(cur, hi)
        if cur < length_m:
            allowed.append((cur, length_m))
        return [(a, b) for a, b in allowed if b > a + 1e-9]

    def _sample_xs(self):
        self.samples_full = []
        self.bed_datum = None
        if self.xs_geom is None or self.dem_layer is None:
            return

        length = self.xs_geom.length()
        if not length or length <= 0.01:
            self._set_notes("XS length too small.")
            return

        step = self._current_step_m()
        if step is None or step <= 0:
            self._set_notes("Invalid sample spacing.")
            return

        provider = self.dem_layer.dataProvider()
        nodata = 0
        n = max(2, int(math.floor(length / step)) + 1)

        for i in range(n):
            d = min(length, i * step)
            pt = self.xs_geom.interpolate(d).asPoint()
            val = provider.sample(QgsPointXY(pt), 1)[0]
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                nodata += 1
                continue
            self.samples_full.append({"x": float(d), "z": float(val)})

        if len(self.samples_full) < 2:
            self._set_notes("DEM sampling failed (too few valid samples).")
            self._plot_profile()
            return

        # chainage direction: if "Reverse" then invert chainage
        if self.cboChainage.currentText().strip().lower().startswith("reverse"):
            length = self.xs_geom.length()
            for s in self.samples_full:
                s["x"] = float(length - s["x"])
            self.samples_full.sort(key=lambda s: s["x"])

        self.bed_datum = min(s["z"] for s in self.samples_full)

        self._excl_ranges = self._parse_exclusions(length)
        self._allowed_ranges = self._compute_allowed_ranges(length, self._excl_ranges)

        warn = ""
        if len(self.samples_full) > 1000:
            warn = f" Warning: XS has {len(self.samples_full)} samples. Plot uses adaptive display."
        self._set_status(f"Status: DEM sampled ({len(self.samples_full)} pts, NoData={nodata}). Bed datum={self.bed_datum:.2f}.{warn}")

        self._plot_profile()

    # ------------- hydraulics -------------
    def _segment_z(self, x1, z1, x2, z2, x):
        if abs(x2 - x1) < 1e-12:
            return z1
        t = (x - x1) / (x2 - x1)
        return z1 + t * (z2 - z1)

    def _iter_allowed_subsegments(self, x1, x2):
        if x2 < x1:
            x1, x2 = x2, x1
        for a, b in self._allowed_ranges:
            lo = max(x1, a)
            hi = min(x2, b)
            if hi > lo + 1e-12:
                yield lo, hi

    def _compute_AP_wetted(self, wsl):
        if not self.samples_full or wsl is None:
            return None, None, [], []

        xs = [s["x"] for s in self.samples_full]
        zs = [s["z"] for s in self.samples_full]
        L = xs[-1]

        A = 0.0
        P = 0.0
        wetted_segments = []
        crossings = []

        for i in range(len(xs) - 1):
            x1, z1 = xs[i], zs[i]
            x2, z2 = xs[i+1], zs[i+1]

            d1 = wsl - z1
            d2 = wsl - z2
            below1 = d1 > 0
            below2 = d2 > 0
            if below1 != below2:
                if abs(z2 - z1) < 1e-12:
                    t = 0.5
                else:
                    t = (wsl - z1) / (z2 - z1)
                t = max(0.0, min(1.0, t))
                xi = x1 + t * (x2 - x1)
                crossings.append(xi)

            for xa, xb in self._iter_allowed_subsegments(x1, x2):
                za = self._segment_z(x1, z1, x2, z2, xa)
                zb = self._segment_z(x1, z1, x2, z2, xb)
                da = max(0.0, wsl - za)
                db = max(0.0, wsl - zb)
                if da <= 0 and db <= 0:
                    continue
                dx = xb - xa
                A += 0.5 * (da + db) * dx

                if da > 0 and db > 0:
                    P += math.hypot(dx, zb - za)
                    wetted_segments.append(((xa, za), (xb, zb)))
                else:
                    if abs(zb - za) < 1e-12:
                        continue
                    tw = (wsl - za) / (zb - za)
                    tw = max(0.0, min(1.0, tw))
                    xw = xa + tw * (xb - xa)
                    zw = wsl
                    if da > 0 and db <= 0:
                        P += math.hypot(xw - xa, zw - za)
                        wetted_segments.append(((xa, za), (xw, zw)))
                    elif db > 0 and da <= 0:
                        P += math.hypot(xb - xw, zb - zw)
                        wetted_segments.append(((xw, zw), (xb, zb)))

        crossings.sort()
        # Build wetted intervals directly from wetted segments (robust for multiple pools and "vertical wall" closures).
        intervals = []
        for seg in wetted_segments:
            (x1, _z1), (x2, _z2) = seg
            lo = min(x1, x2)
            hi = max(x1, x2)
            if hi > lo + 1e-9:
                intervals.append((lo, hi))
        intervals.sort()
        merged = []
        for lo, hi in intervals:
            if not merged:
                merged.append([lo, hi])
            else:
                plo, phi = merged[-1]
                if lo <= phi + 1e-6:
                    merged[-1][1] = max(phi, hi)
                else:
                    merged.append([lo, hi])
        intervals = [(a, b) for a, b in merged]
        return A, P, wetted_segments, intervals

    def _manning_Q(self, A, P, S, n):
        if A is None or P is None or A <= 0 or P <= 0 or S <= 0 or n <= 0:
            return 0.0
        R = A / P
        return (1.0 / n) * A * (R ** (2.0 / 3.0)) * math.sqrt(S)

    def _solve_depth_for_Q(self, Q_target, S, n):
        if self.bed_datum is None:
            return None, "No bed datum."
        if Q_target <= 0:
            return 0.0, None

        def Q_of_depth(d):
            wsl = self.bed_datum + d
            A, P, _, _ = self._compute_AP_wetted(wsl)
            return self._manning_Q(A, P, S, n)

        d_lo = 0.0
        d_hi = 0.5
        Q_hi = Q_of_depth(d_hi)
        max_depth = max(5.0, (max(s["z"] for s in self.samples_full) - self.bed_datum) + 10.0)

        grow = 0
        while Q_hi < Q_target and d_hi < max_depth and grow < 30:
            d_hi *= 2.0
            Q_hi = Q_of_depth(d_hi)
            grow += 1

        if Q_hi < Q_target:
            return None, "Could not bracket depth for target Q (exclusions may block conveyance)."

        for _ in range(60):
            d_mid = 0.5*(d_lo + d_hi)
            Q_mid = Q_of_depth(d_mid)
            if abs(Q_mid - Q_target) <= max(1e-6, 1e-4 * Q_target):
                return d_mid, None
            if Q_mid < Q_target:
                d_lo = d_mid
            else:
                d_hi = d_mid
        return 0.5*(d_lo + d_hi), "Reached solver iteration limit."

    # ------------- crest (exclusion-aware) -------------
    def _is_excluded_x(self, x):
        for a, b in self._excl_ranges:
            if a <= x <= b:
                return True
        return False

    def _estimate_crests(self):
        if not self.samples_full:
            return None, None, None, "No samples."
        xs = [s["x"] for s in self.samples_full]
        zs = [s["z"] for s in self.samples_full]
        L = xs[-1] - xs[0]
        if L <= 0:
            return None, None, None, "XS length invalid."
        end_win = min(20.0, 0.10 * L)
        end_win = max(end_win, 5.0)

        mid_lo = xs[0] + 0.25 * L
        mid_hi = xs[0] + 0.75 * L
        mid_z = [z for x, z in zip(xs, zs) if (mid_lo <= x <= mid_hi and not self._is_excluded_x(x))]
        if not mid_z:
            mid_z = [z for x, z in zip(xs, zs) if not self._is_excluded_x(x)]
        if not mid_z:
            return None, None, None, "Crest not available (exclusions remove all points)."
        mid_ref = sorted(mid_z)[len(mid_z)//2]

        left_z = [z for x, z in zip(xs, zs) if (x <= xs[0] + end_win and not self._is_excluded_x(x))]
        right_z = [z for x, z in zip(xs, zs) if (x >= xs[-1] - end_win and not self._is_excluded_x(x))]

        rise_min = 0.3
        crest_L = max(left_z) if left_z and (max(left_z) - mid_ref) >= rise_min else None
        crest_R = max(right_z) if right_z and (max(right_z) - mid_ref) >= rise_min else None

        notes = []
        if crest_L is None and crest_R is None:
            return None, None, None, "No embankments detected (or excluded)."
        if crest_L is None and crest_R is not None:
            crest_L = crest_R
            notes.append("Left crest assumed (vertical wall).")
        if crest_R is None and crest_L is not None:
            crest_R = crest_L
            notes.append("Right crest assumed (vertical wall).")
        crest_low = min(crest_L, crest_R)
        return crest_L, crest_R, crest_low, (" ".join(notes) if notes else None)

    # ------------- main update -------------
    def on_params_changed(self):
        if self.xs_geom is None:
            self._clear_results()
            self._set_notes("No XS.")
            self._plot_profile()
            return
        if self.dem_layer is None:
            self._clear_results()
            self._set_notes("No DEM selected.")
            self._plot_profile()
            return
        if not self.samples_full:
            self._sample_xs()
            if not self.samples_full:
                self._clear_results()
                self._set_notes("No profile samples.")
                return

        L = self.samples_full[-1]["x"]
        self._excl_ranges = self._parse_exclusions(L)
        self._allowed_ranges = self._compute_allowed_ranges(L, self._excl_ranges)

        S = self._parse_float(self.edtSlopeUser.text())
        n = self._parse_float(self.edtN.text())
        if S is None or S <= 0:
            self._clear_results()
            self._set_notes("S must be > 0.")
            return
        if n is None or n <= 0:
            self._clear_results()
            self._set_notes("n must be > 0.")
            return

        depth_mode = self.rdoInputDepthOutputQ.isChecked()
        if depth_mode:
            depth_in = self._parse_float(self.edtDepth.text())
            if depth_in is None or depth_in < 0:
                self._clear_results()
                self._set_notes("Depth must be >= 0.")
                return
            wsl = self.bed_datum + depth_in
            A, P, wetted_segments, pools = self._compute_AP_wetted(wsl)
            Q = self._manning_Q(A, P, S, n)
            self._update_outputs(Q=Q, depth=depth_in, wsl=wsl, A=A, P=P,
                                 wetted_segments=wetted_segments, pools=pools, solver_note=None)
        else:
            Q_target = self._parse_float(self.edtQ.text())
            if Q_target is None or Q_target < 0:
                self._clear_results()
                self._set_notes("Q must be >= 0.")
                return
            depth, warn = self._solve_depth_for_Q(Q_target, S, n)
            if depth is None:
                self._clear_results()
                self._set_notes(warn or "Solver failed.")
                self._plot_profile()
                return
            wsl = self.bed_datum + depth
            A, P, wetted_segments, pools = self._compute_AP_wetted(wsl)
            self._update_outputs(Q=Q_target, depth=depth, wsl=wsl, A=A, P=P,
                                 wetted_segments=wetted_segments, pools=pools, solver_note=warn)

    def _update_outputs(self, Q, depth, wsl, A, P, wetted_segments, pools, solver_note=None):
        if A is None or P is None or A <= 0 or P <= 0:
            self._clear_results()
            self._set_notes("WSL below bed or invalid wetted geometry.")
            self._plot_profile(wsl=wsl)
            return

        R = A / P
        V = Q / A if A > 0 else 0.0

        self.edtQOut.setText(f"{Q:.6g}")
        self.edtDepthOut.setText(f"{depth:.6g}")
        self.edtWSL.setText(f"{wsl:.6g}")
        self.edtA.setText(f"{A:.6g}")
        self.edtP.setText(f"{P:.6g}")
        self.edtR.setText(f"{R:.6g}")
        self.edtV.setText(f"{V:.6g}")
        if hasattr(self, "lblPOverlay"):
            self.lblPOverlay.setText(f"P = {P:.3f} m")

        crest_L, crest_R, crest_low, crest_note = self._estimate_crests()
        if crest_low is not None:
            self.edtCrestLow.setText(f"{crest_low:.6g}")
            freeboard = crest_low - wsl
            self.edtFreeboard.setText(f"{freeboard:.6g}")
        else:
            self.edtCrestLow.setText("--")
            self.edtFreeboard.setText("--")

        notes = []
        if solver_note:
            notes.append(solver_note)
        if crest_note:
            notes.append(crest_note)
        if self._excl_ranges:
            notes.append("Crest recalculated with exclusions.")
        if len(self.samples_full) > 1000:
            notes.append(f"XS has {len(self.samples_full)} pts; adaptive plot on.")
        self._set_notes(" ".join(notes) if notes else "--")

        self._plot_profile(wsl=wsl, wetted_segments=wetted_segments, pools=pools, crest_low=crest_low)

    # ------------- plot (adaptive + features) -------------
    def _init_plot(self):
        if not _HAS_MPL:
            return
        fig = Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlabel("Chainage (m)")
        ax.set_ylabel("Elevation (m)")
        canvas = FigureCanvas(fig)
        self._mpl_canvas = canvas
        self._mpl_ax = ax
        layout = QtWidgets.QVBoxLayout(self.plotPlaceholder)
        layout.setContentsMargins(0, 0, 0, 0)
        # Keep a hidden toolbar for navigation actions, but do not show buttons in the UI.
        self._mpl_toolbar = NavigationToolbar(canvas, self)
        self._mpl_toolbar.hide()
        layout.addWidget(canvas)
        # Right-click menu for plot controls
        canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        canvas.customContextMenuRequested.connect(self._show_plot_menu)
        self._reset_plot_tools()

        self._artists["crosshair_v"] = ax.axvline(0, linewidth=0.8, alpha=0.4, visible=False)
        self._artists["crosshair_h"] = ax.axhline(0, linewidth=0.8, alpha=0.4, visible=False)
        self._artists["snap_pt"] = ax.plot([], [], marker="o", markersize=5, linestyle="None", visible=False)[0]
        self._artists["readout"] = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left", fontsize=9, visible=False)

        canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        canvas.mpl_connect("button_press_event", self._on_plot_click)
        canvas.mpl_connect("axes_leave_event", self._on_axes_leave)

    def _adaptive_decimate(self, xs, zs):
        n = len(xs)
        if n <= 1000:
            return xs, zs
        step = max(1, n // 1500)
        return xs[::step], zs[::step]

    def _plot_profile(self, wsl=None, wetted_segments=None, pools=None, crest_low=None):
        self._last_plot_args = {"wsl": wsl, "wetted_segments": wetted_segments, "pools": pools, "crest_low": crest_low}
        if not _HAS_MPL or self._mpl_ax is None:
            return
        ax = self._mpl_ax
        ax.clear()
        ax.set_xlabel("Chainage (m)")
        ax.set_ylabel("Elevation (m)")
        self._artists.clear()

        if not self.samples_full:
            self._mpl_canvas.draw_idle()
            self._reset_plot_tools()
            return

        xs = [s["x"] for s in self.samples_full]
        zs = [s["z"] for s in self.samples_full]
        dxs, dzs = self._adaptive_decimate(xs, zs)

        xmin = min(xs)
        xmax = max(xs)
        if self._plot_flags.get("show_bed", True):
            ax.plot(dxs, dzs, linewidth=1.5, color="black", label="Bed")

        # exclusion shading
        if self._plot_flags.get("show_excl", True) and self._excl_ranges:
            for a, b in self._excl_ranges:
                ax.axvspan(a, b, alpha=0.08)

        # WSL
        if wsl is not None and self._plot_flags.get("show_wsl", True):
            ax.plot([xmin, xmax], [wsl, wsl], linewidth=1.2, color="#69b3ff", label="WSL (m)")

        # Crest low
        if crest_low is not None and self._plot_flags.get("show_crest", True):
            first = True
            for a, b in self._allowed_ranges:
                if b <= a:
                    continue
                ax.plot([a, b], [crest_low, crest_low], linewidth=1.0, linestyle="--", color="green", label=("Crest (low)" if first else "_nolegend_"))
                first = False

        # Wetted perimeter highlight
        if wetted_segments and self._plot_flags.get("show_p", True):
            xp = []
            yp = []
            nan = float("nan")
            for seg in wetted_segments:
                (x1, z1), (x2, z2) = seg
                xp.extend([x1, x2, nan])
                yp.extend([z1, z2, nan])
            ax.plot(xp, yp, linewidth=2.5, linestyle="--", color="#006400", label="P (m)")

        # Hatched wetted area for pools + exclusions
        if (wsl is not None) and self._plot_flags.get("show_a", True):
            self._draw_area_hatch_polygons(ax, wsl, pools)

        # crosshair/readout
        self._artists["crosshair_v"] = ax.axvline(0, linewidth=0.8, alpha=0.35, color="0.4", visible=False, label="_nolegend_")
        self._artists["crosshair_h"] = ax.axhline(0, linewidth=0.8, alpha=0.35, color="0.4", visible=False, label="_nolegend_")
        self._artists["snap_pt"] = ax.plot([], [], marker="o", markersize=5, linestyle="None", color="0.4", visible=False, label="_nolegend_")[0]
        self._artists["readout"] = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left", fontsize=9, visible=False)

        ax.legend(loc="best", fontsize=9)
        # Tight y-limits so the profile doesn't look "flat"
        y_candidates = list(dzs)
        if wsl is not None and self._plot_flags.get("show_wsl", True):
            y_candidates.append(wsl)
        if crest_low is not None and self._plot_flags.get("show_crest", True):
            y_candidates.append(crest_low)
        y_min = min(y_candidates) if y_candidates else min(dzs)
        y_max = max(y_candidates) if y_candidates else max(dzs)
        yr = max(1e-6, y_max - y_min)
        pad = max(0.2, 0.05 * yr)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlim(xmin, xmax)
        self._mpl_canvas.draw_idle()

    def _draw_area_hatch_polygons(self, ax, wsl, intervals):
        if not intervals:
            intervals = []

        npts = len(self.samples_full) if getattr(self, "samples_full", None) else 0
        mode = getattr(self, "_plot_quality", "adaptive")
        if mode == "adaptive":
            if npts <= 1000:
                q = "full"
            elif npts <= 5000:
                q = "mid"
            else:
                q = "light"
        elif mode == "full":
            q = "full"
        else:
            q = "light"

        xs = [s["x"] for s in self.samples_full] if self.samples_full else []
        zs = [s["z"] for s in self.samples_full] if self.samples_full else []
        if not xs:
            return

        def bed_z(x):
            if x <= xs[0]:
                return zs[0]
            if x >= xs[-1]:
                return zs[-1]
            lo = 0
            hi = len(xs) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if xs[mid] <= x:
                    lo = mid
                else:
                    hi = mid
            x1, x2 = xs[lo], xs[hi]
            z1, z2 = zs[lo], zs[hi]
            if abs(x2 - x1) < 1e-12:
                return z1
            t = (x - x1) / (x2 - x1)
            return z1 + t * (z2 - z1)

        proxy_done = False

        for (xL, xR) in intervals:
            if xR <= xL + 1e-9:
                continue

            # Apply exclusion/allowed ranges
            for a, b in self._allowed_ranges:
                lo = max(xL, a)
                hi = min(xR, b)
                if hi <= lo + 1e-9:
                    continue

                if q == "full":
                    x_list = [x for x in xs if lo <= x <= hi]
                elif q == "mid":
                    step = max(1, npts // 1500)
                    x_list = [x for i, x in enumerate(xs) if (i % step == 0 and lo <= x <= hi)]
                else:
                    step = max(1, npts // 800)
                    x_list = [x for i, x in enumerate(xs) if (i % step == 0 and lo <= x <= hi)]

                if not x_list or x_list[0] > lo + 1e-9:
                    x_list = [lo] + x_list
                if x_list[-1] < hi - 1e-9:
                    x_list = x_list + [hi]

                y_bed = [min(bed_z(x), wsl) for x in x_list]

                poly_x = x_list + list(reversed(x_list))
                poly_y = y_bed + [wsl for _ in x_list]

                lbl = "A (m2)" if not proxy_done else "_nolegend_"
                proxy_done = True

                ax.fill(poly_x, poly_y, facecolor="none", edgecolor="0.5", hatch="///", linewidth=0.6, label=lbl)

    def _reset_plot_tools(self):
        # Ensure the plot is in "pointer" mode (Pan/Zoom off) so left-click read works.
        if not _HAS_MPL or self._mpl_toolbar is None:
            return
        try:
            mode = getattr(self._mpl_toolbar, "mode", "")
            if mode == "pan/zoom":
                self._mpl_toolbar.pan()
            elif mode == "zoom rect":
                self._mpl_toolbar.zoom()
        except Exception:
            # Best-effort fallback
            try:
                if getattr(self._mpl_toolbar, "mode", "") == "pan/zoom":
                    self._mpl_toolbar.pan()
            except Exception:
                pass
            try:
                if getattr(self._mpl_toolbar, "mode", "") == "zoom rect":
                    self._mpl_toolbar.zoom()
            except Exception:
                pass

    def _show_plot_menu(self, pos):
        if not _HAS_MPL or self._mpl_canvas is None:
            return
        menu = QMenu(self)

        act_home = menu.addAction("Home")
        act_back = menu.addAction("Back")
        act_fwd = menu.addAction("Forward")
        menu.addSeparator()

        act_pan = menu.addAction("Pan")
        act_zoom = menu.addAction("Zoom")
        menu.addSeparator()

        act_save = menu.addAction("Save image...")
        menu.addSeparator()

        sub = menu.addMenu("Plot quality")
        act_adapt = sub.addAction("Adaptive (recommended)")
        act_full = sub.addAction("Full fidelity (force)")
        act_light = sub.addAction("Lightweight (force)")

        npts = len(self.samples_full) if getattr(self, "samples_full", None) else 0
        if npts > 1000:
            act_full.setEnabled(False)

        action = menu.exec_(self._mpl_canvas.mapToGlobal(pos))
        if action is None:
            return

        if action == act_home:
            try:
                self._mpl_toolbar.home()
            except Exception:
                self._plot_profile(**self._last_plot_args)
            return
        if action == act_back:
            try:
                self._mpl_toolbar.back()
            except Exception:
                pass
            return
        if action == act_fwd:
            try:
                self._mpl_toolbar.forward()
            except Exception:
                pass
            return
        if action == act_pan:
            try:
                self._mpl_toolbar.pan()
            except Exception:
                pass
            return
        if action == act_zoom:
            try:
                self._mpl_toolbar.zoom()
            except Exception:
                pass
            return
        if action == act_save:
            try:
                self._mpl_toolbar.save_figure()
            except Exception:
                pass
            return

        if action == act_adapt:
            self._plot_quality = "adaptive"
            self._plot_profile(**self._last_plot_args)
            return
        if action == act_full:
            self._plot_quality = "full"
            self._plot_profile(**self._last_plot_args)
            return
        if action == act_light:
            self._plot_quality = "light"
            self._plot_profile(**self._last_plot_args)
            return

    def _on_scroll_zoom(self, event):
        if event.inaxes != self._mpl_ax:
            return
        ax = self._mpl_ax
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return
        base_scale = 1.2
        scale = 1/base_scale if event.button == "up" else base_scale
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[1] - ylim[0]) * scale
        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0] + 1e-12)
        rely = (ydata - ylim[0]) / (ylim[1] - ylim[0] + 1e-12)
        ax.set_xlim([xdata - new_w * relx, xdata + new_w * (1 - relx)])
        ax.set_ylim([ydata - new_h * rely, ydata + new_h * (1 - rely)])
        self._mpl_canvas.draw_idle()

    def _on_motion(self, event):
        if event.inaxes != self._mpl_ax:
            return
        try:
            import time
            t = time.time()
            if (t - self._last_motion_t) < self._motion_min_dt:
                return
            self._last_motion_t = t
        except Exception:
            pass
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return

        # snap by chainage
        xs = [s["x"] for s in self.samples_full]
        zs = [s["z"] for s in self.samples_full]
        import bisect
        i = bisect.bisect_left(xs, x)
        if i <= 0:
            idx = 0
        elif i >= len(xs):
            idx = len(xs) - 1
        else:
            idx = i if abs(xs[i] - x) < abs(xs[i-1] - x) else i-1
        sx, sz = xs[idx], zs[idx]

        cv = self._artists.get("crosshair_v")
        ch = self._artists.get("crosshair_h")
        sp = self._artists.get("snap_pt")
        ro = self._artists.get("readout")
        if cv:
            cv.set_xdata([sx, sx]); cv.set_visible(True)
        if ch:
            ch.set_ydata([sz, sz]); ch.set_visible(True)
        if sp:
            sp.set_data([sx], [sz]); sp.set_visible(True)
        if ro:
            ro.set_text(f"x={sx:.2f} m\nz={sz:.2f} m")
            ro.set_visible(True)
        self._mpl_canvas.draw_idle()

    def _on_plot_click(self, event):
        # No snapping. Left-click to read X,Y at cursor.
        if event.inaxes != self._mpl_ax:
            return
        if event.button != 1:
            return
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return
        try:
            txt = f"X={x:.3f}, Y={y:.3f}"
        except Exception:
            txt = f"X={x}, Y={y}"
        if hasattr(self, "lblCursor"):
            self.lblCursor.setText(txt)
        elif hasattr(self, "lblStatus"):
            base = self.lblStatus.text().split(" | ")[0]
            self.lblStatus.setText(f"{base} | {txt}")

    def _on_axes_leave(self, event):
        for k in ("crosshair_v","crosshair_h","snap_pt","readout"):
            a = self._artists.get(k)
            if a:
                a.set_visible(False)
        self._mpl_canvas.draw_idle()

    # ------------- label double-click toggles -------------
    def _install_label_toggles(self):
        self._lbl_filter = _LabelDblClickFilter(self._on_label_dblclick)
        self._label_map = {
            "lblP": "show_p",
            "lblA": "show_a",
            "lblWSL": "show_wsl",
            "lblCrestLow": "show_crest",
            "lblFreeboard": "show_crest",
        }
        for obj_name in self._label_map.keys():
            w = getattr(self, obj_name, None)
            if w:
                w.installEventFilter(self._lbl_filter)

    def _on_label_dblclick(self, obj_name):
        key = self._label_map.get(obj_name)
        if not key:
            return
        self._plot_flags[key] = not self._plot_flags.get(key, True)
        wsl = self._parse_float(self.edtWSL.text())
        if wsl is not None and self.samples_full:
            A, P, wetted_segments, pools = self._compute_AP_wetted(wsl)
            _, _, crest_low, _ = self._estimate_crests()
            self._plot_profile(wsl=wsl, wetted_segments=wetted_segments, pools=pools, crest_low=crest_low)
        else:
            self._plot_profile()

    # ------------- Wetted polygon layer -------------
    def on_create_wetted_polygon(self):
        wsl = self._parse_float(self.edtWSL.text())
        A_calc = self._parse_float(self.edtA.text())
        if wsl is None or A_calc is None or not self.samples_full:
            QMessageBox.information(self, "QEP18", "No valid results to build wetted polygon. Run a calculation first.")
            return
        polys = self._build_wetted_polygons_chainage_space(wsl)
        if not polys:
            QMessageBox.information(self, "QEP18", "Could not build wetted polygon (no wetted area found).")
            return

        layer = QgsVectorLayer("MultiPolygon?crs=EPSG:3857", "QEP18_WettedArea_XS", "memory")
        prov = layer.dataProvider()
        prov.addAttributes([
            QgsField("WSL", 20),
            QgsField("A_calc", 20),
            QgsField("Q", 20),
            QgsField("Depth", 20),
            QgsField("exclude", 120),
        ])
        layer.updateFields()

        feat = QgsFeature(layer.fields())
        feat.setGeometry(QgsGeometry.fromMultiPolygonXY(polys))
        feat["WSL"] = float(wsl)
        feat["A_calc"] = float(A_calc)
        qv = self._parse_float(self.edtQOut.text()) or 0.0
        dv = self._parse_float(self.edtDepthOut.text()) or 0.0
        feat["Q"] = float(qv)
        feat["Depth"] = float(dv)
        feat["exclude"] = (self.edtExcludeRanges.text() or "").strip()
        prov.addFeatures([feat])
        layer.updateExtents()
        QgsProject.instance().addMapLayer(layer)

        QMessageBox.information(self, "QEP18",
            "Wetted polygon layer created: 'QEP18_WettedArea_XS'.\n"
            "Note: Geometry is in chainage-elevation space for $area cross-check (not map space)."
        )

    def _build_wetted_polygons_chainage_space(self, wsl):
        xs = [s["x"] for s in self.samples_full]
        zs = [s["z"] for s in self.samples_full]
        L = xs[-1]

        aug = []
        for i in range(len(xs) - 1):
            x1, z1 = xs[i], zs[i]
            x2, z2 = xs[i+1], zs[i+1]
            if i == 0:
                aug.append((x1, z1))
            d1 = wsl - z1
            d2 = wsl - z2
            below1 = d1 > 0
            below2 = d2 > 0
            if below1 != below2:
                if abs(z2 - z1) < 1e-12:
                    t = 0.5
                else:
                    t = (wsl - z1) / (z2 - z1)
                t = max(0.0, min(1.0, t))
                xi = x1 + t * (x2 - x1)
                aug.append((xi, wsl))
            aug.append((x2, z2))

        crossing_idx = [i for i, (_, z) in enumerate(aug) if abs(z - wsl) < 1e-9]
        polys = []

        if len(crossing_idx) < 2:
            if any(z < wsl for (_, z) in aug):
                ring = []
                ring.append(QgsPointXY(0.0, wsl))
                for (x, z) in aug:
                    if self._is_excluded_x(x):
                        continue
                    ring.append(QgsPointXY(x, z))
                ring.append(QgsPointXY(L, wsl))
                ring.append(QgsPointXY(0.0, wsl))
                if len(ring) >= 4:
                    polys.append([ring])
            return polys

        for k in range(0, len(crossing_idx)-1, 2):
            iL = crossing_idx[k]
            iR = crossing_idx[k+1]
            if iR <= iL:
                continue
            xL, _ = aug[iL]
            xR, _ = aug[iR]

            for a, b in self._allowed_ranges:
                lo = max(xL, a)
                hi = min(xR, b)
                if hi <= lo + 1e-9:
                    continue
                ring = []
                ring.append(QgsPointXY(lo, wsl))
                for i in range(iL, iR+1):
                    x, z = aug[i]
                    if x < lo - 1e-9 or x > hi + 1e-9:
                        continue
                    if self._is_excluded_x(x):
                        continue
                    ring.append(QgsPointXY(x, z))
                ring.append(QgsPointXY(hi, wsl))
                ring.append(QgsPointXY(lo, wsl))
                if len(ring) >= 4:
                    polys.append([ring])
        return polys

    # ------------- Export CSV -------------
    def on_export_csv(self):
        if not self.samples_full:
            QMessageBox.information(self, "QEP18", "No profile to export. Draw/select an XS and sample a DEM first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV (*.csv)")
        if not path:
            return

        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dem": self.dem_layer.name() if self.dem_layer else "",
            "sample_spacing": self._spacing_text_for_save(),
            "exclude": (self.edtExcludeRanges.text() or "").strip(),
            "mode": "Depth->Q" if self.rdoInputDepthOutputQ.isChecked() else "Q->Depth",
            "S": self.edtSlopeUser.text(),
            "n": self.edtN.text(),
            "Q_out": self.edtQOut.text(),
            "Depth_out": self.edtDepthOut.text(),
            "WSL": self.edtWSL.text(),
            "A": self.edtA.text(),
            "P": self.edtP.text(),
            "R": self.edtR.text(),
            "V": self.edtV.text(),
            "Crest_low": self.edtCrestLow.text(),
            "Freeboard": self.edtFreeboard.text(),
        }

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["QFAT01 Flow Estimator Export"])
                w.writerow([])
                w.writerow(["Summary"])
                for k, v in summary.items():
                    w.writerow([k, v])
                w.writerow([])
                w.writerow(["Profile"])
                w.writerow(["chainage_m", "elev_m", "excluded"])
                for s in self.samples_full:
                    ex = 1 if self._is_excluded_x(s["x"]) else 0
                    w.writerow([f"{s['x']:.6g}", f"{s['z']:.6g}", ex])
            QMessageBox.information(self, "QEP18", "CSV exported.")
        except Exception as e:
            QMessageBox.warning(self, "QEP18", f"Failed to export CSV:\n{e}")

    # ------------- Copy Summary -------------
    def on_copy_summary(self):
        txt = (
            f"Q={self.edtQOut.text()} m3/s, Depth={self.edtDepthOut.text()} m, WSL={self.edtWSL.text()} mAHD, "
            f"A={self.edtA.text()} m2, P={self.edtP.text()} m, R={self.edtR.text()} m, V={self.edtV.text()} m/s, "
            f"S={self.edtSlopeUser.text()}, n={self.edtN.text()}, Freeboard={self.edtFreeboard.text()} m, "
            f"Exclude={(self.edtExcludeRanges.text() or '').strip()}"
        )
        QtWidgets.QApplication.clipboard().setText(txt)
        self._set_notes("Summary copied to clipboard.")
