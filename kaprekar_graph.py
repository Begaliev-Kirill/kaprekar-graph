#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Граф функции Капрекара в системе счисления с основанием p.

f(n) = (цифры n по убыванию) - (цифры n по возрастанию); запись n всегда
дополняется нулями слева до q разрядов. Пример: f(1001) = 1100 - 0011 = 1089.

Запуск:  python kaprekar_graph.py

Управление: колесо мыши - масштаб, левая кнопка + движение - перемещение,
клик по вершине - путь до цикла, поле «Найти число» - переход к числу.
"""
import math
import random
import sys
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_NODES = 250_000   # жёсткий потолок числа вершин
LABEL_MAX = 250      # подписи рисуются, только если в кадре не больше стольких вершин
BG = "#12141a"


# ════════════════════════════ математика ════════════════════════════
def kaprekar(n, p, q):
    """Функция Капрекара для q-значного числа n в системе с основанием p."""
    d = []
    for _ in range(q):
        n, r = divmod(n, p)
        d.append(r)
    d.sort()
    lo = hi = 0
    for x in d:
        lo = lo * p + x
    for x in reversed(d):
        hi = hi * p + x
    return hi - lo


def to_base(n, p, q):
    """Запись n в системе с основанием p ровно из q цифр (с ведущими нулями)."""
    out = []
    for _ in range(q):
        n, r = divmod(n, p)
        out.append(DIGITS[r])
    return "".join(reversed(out))


def build_graph(p, q, limit, randomize=False):
    """
    Строит граф не более чем из `limit` вершин.
    Берём стартовые числа (подряд или случайно), пропускаем «репдиджиты»
    (все цифры равны) и добавляем каждое число вместе с цепочкой его образов
    до уже известной вершины.
    Возвращает (values, nxt): values[i] - число, nxt[i] - индекс f(values[i]).
    """
    total = p ** q
    cand = min(total, 2 * limit + 2 * p + 16)
    if randomize and total > cand:
        if total < 2 ** 62:
            starts = random.sample(range(total), cand)
        else:
            starts = [random.randrange(total) for _ in range(cand)]
    else:
        starts = range(cand)

    index, values, nxt = {}, [], []
    for s in starts:
        if s in index:
            continue
        room = limit - len(values)
        if room <= 0:
            break
        y = kaprekar(s, p, q)
        if y == 0:                       # у стартового числа все цифры равны
            continue
        chain, imgs, seen = [s], [y], {s}
        x, overflow = y, False
        while x not in index and x not in seen:
            if len(chain) >= room:
                overflow = True
                break
            seen.add(x)
            chain.append(x)
            x = kaprekar(x, p, q)
            imgs.append(x)
        if overflow:
            continue
        base = len(values)
        for j, v in enumerate(chain):
            index[v] = base + j
        values.extend(chain)
        nxt.extend(index[v] for v in imgs)
    return values, np.array(nxt, dtype=np.int64)


def find_cycles(nxt):
    """Циклы функционального графа."""
    n = len(nxt)
    state = [0] * n
    cycles = []
    for s in range(n):
        if state[s]:
            continue
        path, x = [], s
        while state[x] == 0:
            state[x] = 1
            path.append(x)
            x = nxt[x]
        if state[x] == 1:
            cycles.append(path[path.index(x):])
        for v in path:
            state[v] = 2
    return cycles


# ════════════════════════════ раскладка ════════════════════════════
GOLDEN = math.pi * (3 - math.sqrt(5))


def _ring(radii, floor, gap):
    r = [x + gap / 2 for x in radii]
    rho = max(floor, max(r), sum(r) / math.pi)
    while True:
        w = [2 * math.asin(min(1.0, x / rho)) for x in r]
        tot = sum(w)
        if tot <= 2 * math.pi:
            break
        rho *= max(1.0005, tot / (2 * math.pi))
    slack = (2 * math.pi - tot) / len(r)
    ang, acc = [], 0.0
    for x in w:
        ang.append(acc + x / 2)
        acc += x + slack
    return rho, ang


def _shells(radii, base, gap):
    r = [x + gap / 2 for x in radii]
    out = [None] * len(r)
    i = 0
    while i < len(r):
        j, tot, ws = i, 0.0, []
        while j < len(r):
            w = 2 * math.asin(r[j] / (base + r[j]))
            if ws and tot + w > 2 * math.pi:
                break
            ws.append(w)
            tot += w
            j += 1
        slack = max(0.0, 2 * math.pi - tot) / len(ws)
        acc = 0.0
        for k, w in enumerate(ws):
            out[i + k] = (base + r[i + k], acc + w / 2)
            acc += w + slack
        base += 2 * r[j - 1]
        i = j
    return out


def _enclose(discs):
    cx = (min(x - r for x, y, r in discs) + max(x + r for x, y, r in discs)) / 2
    cy = (min(y - r for x, y, r in discs) + max(y + r for x, y, r in discs)) / 2
    return cx, cy, max(math.hypot(x - cx, y - cy) + r for x, y, r in discs)


def layout(nxt, cycles, gap=0.8):
    n = len(nxt)
    nxt_np = np.asarray(nxt)
    by_target = np.argsort(nxt_np, kind="stable").tolist()
    offs = np.concatenate(([0], np.cumsum(np.bincount(nxt_np, minlength=n)))).tolist()
    on_cycle = [False] * n
    for c in cycles:
        for v in c:
            on_cycle[v] = True
    depth, par = [0] * n, [-1] * n
    R, CX, CY = [0.0] * n, [0.0] * n, [0.0] * n
    ox, oy, rot = [0.0] * n, [0.0] * n, [0.0] * n
    X, Y, TH = [0.0] * n, [0.0] * n, [0.0] * n

    def place(u, dist, a):
        ca, sa = math.cos(a), math.sin(a)
        d = dist - math.hypot(CX[u], CY[u])
        ox[u], oy[u] = d * ca, d * sa
        rot[u] = a - math.atan2(CY[u], CX[u])
        return dist * ca, dist * sa, R[u]

    blocks = []
    for cyc in cycles:
        pre, kids_of = [], {}
        for c in cyc:
            stack = [c]
            while stack:
                v = stack.pop()
                pre.append(v)
                ks = [u for u in by_target[offs[v]:offs[v + 1]] if not on_cycle[u]]
                kids_of[v] = ks
                for u in ks:
                    depth[u] = depth[v] + 1
                    par[u] = v
                    stack.append(u)

        for v in reversed(pre):
            ks = kids_of[v]
            leaves = [u for u in ks if not kids_of[u]]
            inner = sorted((u for u in ks if kids_of[u]), key=lambda u: R[u])
            for j, u in enumerate(leaves):
                r, a = math.sqrt(1.0 + 0.4 * (j + 1)), j * GOLDEN
                ox[u], oy[u] = r * math.cos(a), r * math.sin(a)
            rl = (math.sqrt(1.0 + 0.4 * len(leaves)) if leaves else 0.0) + 0.6
            discs = [(0.0, 0.0, rl)]
            if inner:
                spots = _shells([R[u] for u in inner], rl + gap / 2, gap)
                discs += [place(u, dist, a) for u, (dist, a) in zip(inner, spots)]
            CX[v], CY[v], R[v] = _enclose(discs)

        if len(cyc) == 1:
            cx, cy, rc = CX[cyc[0]], CY[cyc[0]], R[cyc[0]]
        else:
            rho, ang = _ring([R[c] for c in cyc], 0.0, gap)
            cx, cy, rc = _enclose([place(c, rho, a) for c, a in zip(cyc, ang)])
        for v in pre:
            p_ = par[v]
            if p_ < 0:
                X[v], Y[v], TH[v] = ox[v], oy[v], rot[v]
            else:
                ct, st = math.cos(TH[p_]), math.sin(TH[p_])
                X[v] = X[p_] + ct * ox[v] - st * oy[v]
                Y[v] = Y[p_] + st * ox[v] + ct * oy[v]
                TH[v] = TH[p_] + rot[v]
        blocks.append((np.array(pre), cx, cy, rc))

    Xa, Ya = np.array(X), np.array(Y)
    radii = [b[3] for b in blocks]
    order = sorted(range(len(blocks)), key=lambda i: -radii[i])
    side = max(math.sqrt(sum((2 * r + 2) ** 2 for r in radii)) * 1.15,
               2 * radii[order[0]] + 2)
    pos = np.zeros((n, 2))
    x = y = rowh = 0.0
    for i in order:
        w = 2 * radii[i] + 2
        if x > 0 and x + w > side:
            x, y, rowh = 0.0, y + rowh, 0.0
        idx, cx, cy, _ = blocks[i]
        pos[idx, 0] = Xa[idx] - cx + x + w / 2
        pos[idx, 1] = Ya[idx] - cy + y + w / 2
        x += w
        rowh = max(rowh, w)
    return pos, np.array(depth, dtype=np.int32), [len(b[0]) for b in blocks]


def edge_geometry(pos, nxt):
    nxt = np.asarray(nxt)
    src = np.flatnonzero(nxt != np.arange(len(nxt)))
    a, b = pos[src], pos[nxt[src]]
    seg = np.empty((2 * len(src), 2))
    seg[0::2], seg[1::2] = a, b
    v = b - a
    length = np.hypot(v[:, 0], v[:, 1]) + 1e-12
    u = v / length[:, None]
    nrm = np.stack((-u[:, 1], u[:, 0]), axis=1)
    h = np.minimum(0.55, 0.3 * length)[:, None]
    tip = a + v * 0.62
    base = tip - u * h
    head = np.empty((4 * len(src), 2))
    head[0::4], head[1::4] = tip, base + nrm * 0.4 * h
    head[2::4], head[3::4] = tip, base - nrm * 0.4 * h
    return seg, head


def make_palette(maxd):
    pal = [pg.mkBrush(255, 196, 61)]
    for d in range(1, maxd + 1):
        t = (d - 1) / max(1, maxd - 1)
        pal.append(pg.mkBrush(pg.hsvColor(0.52 + 0.40 * t, 0.62, 1.0)))
    return pal


# ════════════════════════════ интерфейс ════════════════════════════
class KaprekarWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Граф функции Капрекара")
        self.resize(1320, 840)
        self.n = 0
        self.p = self.q = 0
        self.values, self.index, self.nxt = [], {}, []
        self.depth = np.zeros(0, dtype=np.int32)
        self.indeg = np.zeros(0, dtype=np.int64)
        self.px = self.py = np.zeros(0)
        self.node_items, self.label_pool, self.label_idx = [], [], []

        self._build_ui()

        self.label_timer = QtCore.QTimer(self)
        self.label_timer.setSingleShot(True)
        self.label_timer.setInterval(70)
        self.label_timer.timeout.connect(self.update_labels)
        self.vb.sigRangeChanged.connect(lambda *_: self.label_timer.start())
        self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=40, slot=self.on_move)
        self.plot.scene().sigMouseClicked.connect(self.on_click)
        QtCore.QTimer.singleShot(0, self.rebuild)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        side = QtWidgets.QWidget()
        side.setFixedWidth(330)
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)

        title = QtWidgets.QLabel("Функция Капрекара")
        title.setStyleSheet("font-size: 17px; font-weight: 600;")
        sl.addWidget(title)

        form = QtWidgets.QFormLayout()
        self.sp_p = QtWidgets.QSpinBox()
        self.sp_p.setRange(2, 36)
        self.sp_p.setValue(10)
        self.sp_q = QtWidgets.QSpinBox()
        self.sp_q.setRange(2, 24)
        self.sp_q.setValue(4)
        self.sp_n = QtWidgets.QSpinBox()
        self.sp_n.setRange(10, MAX_NODES)
        self.sp_n.setSingleStep(100)
        self.sp_n.setValue(MAX_NODES)
        self.sl_n = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_n.setRange(10, MAX_NODES)
        self.sl_n.setValue(MAX_NODES)
        self.sl_n.valueChanged.connect(self.sp_n.setValue)
        self.sp_n.valueChanged.connect(self.sl_n.setValue)
        form.addRow("Основание p:", self.sp_p)
        form.addRow("Длина записи q:", self.sp_q)
        form.addRow("Макс. чисел:", self.sp_n)
        form.addRow("", self.sl_n)
        sl.addLayout(form)

        self.lb_total = QtWidgets.QLabel()
        self.lb_total.setStyleSheet("color: #8d96a6;")
        sl.addWidget(self.lb_total)
        self.sp_p.valueChanged.connect(self.update_total)
        self.sp_q.valueChanged.connect(self.update_total)
        self.update_total()

        self.chk_rand = QtWidgets.QCheckBox("Случайная выборка чисел")
        self.chk_rand.setToolTip("Иначе берутся числа подряд: 0, 1, 2, ... (если p^q больше лимита)")
        self.chk_arrows = QtWidgets.QCheckBox("Стрелки на рёбрах")
        self.chk_arrows.setChecked(True)
        self.chk_aa = QtWidgets.QCheckBox("Сглаживание линий (медленнее)")
        self.chk_arrows.toggled.connect(lambda on: self.heads.setVisible(on))
        self.chk_aa.toggled.connect(self.set_antialias)
        for c in (self.chk_rand, self.chk_arrows, self.chk_aa):
            sl.addWidget(c)

        btns = QtWidgets.QHBoxLayout()
        self.btn_build = QtWidgets.QPushButton("Построить граф")
        self.btn_build.setStyleSheet("font-weight: 600; padding: 6px;")
        self.btn_build.clicked.connect(self.rebuild)
        self.btn_fit = QtWidgets.QPushButton("Показать всё")
        self.btn_fit.setStyleSheet("padding: 6px;")
        self.btn_fit.clicked.connect(self.fit)
        btns.addWidget(self.btn_build)
        btns.addWidget(self.btn_fit)
        sl.addLayout(btns)

        self.ed_search = QtWidgets.QLineEdit()
        self.ed_search.setPlaceholderText("Найти число (в системе p) и нажать Enter")
        self.ed_search.returnPressed.connect(self.on_search)
        sl.addWidget(self.ed_search)

        self.lb_legend = QtWidgets.QLabel()
        self.lb_legend.setWordWrap(True)
        sl.addWidget(self.lb_legend)

        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setFont(QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont))
        sl.addWidget(self.txt, 1)

        pg.setConfigOptions(antialias=False)
        self.plot = pg.PlotWidget(background=BG)
        pi = self.plot.getPlotItem()
        pi.hideAxis("left")
        pi.hideAxis("bottom")
        pi.hideButtons()
        pi.setMenuEnabled(False)
        self.vb = pi.getViewBox()
        self.vb.setAspectLocked(True)

        self.edges = pg.PlotCurveItem(pen=pg.mkPen((120, 135, 165, 90), width=1),
                                       skipFiniteCheck=True)
        self.heads = pg.PlotCurveItem(pen=pg.mkPen((235, 240, 255, 150), width=1),
                                       skipFiniteCheck=True)
        self.hl_line = pg.PlotCurveItem(pen=pg.mkPen((255, 214, 90), width=3))
        self.hl_nodes = pg.ScatterPlotItem(size=13, pxMode=True,
                                            pen=pg.mkPen("w", width=2),
                                            brush=pg.mkBrush(0, 0, 0, 0))
        for z, it, ib in ((1, self.edges, False), (2, self.heads, False),
                          (4, self.hl_line, True), (5, self.hl_nodes, True)):
            it.setZValue(z)
            self.vb.addItem(it, ignoreBounds=ib)

        root.addWidget(side)
        root.addWidget(self.plot, 1)
        self.statusBar().showMessage(
            "Колесо - масштаб, перетаскивание - перемещение, клик по вершине - путь")

    def update_total(self):
        t = self.sp_p.value() ** self.sp_q.value()
        txt = f"{t:,}".replace(",", " ") if t < 10 ** 12 else f"{t:.2e}"
        self.lb_total.setText(f"Всего q-значных чисел p^q = {txt}")

    def set_antialias(self, on):
        for it in (self.edges, self.heads, self.hl_line):
            it.opts["antialias"] = bool(on)
            it.update()

    def lab(self, i):
        return to_base(self.values[i], self.p, self.q)

    def fit(self):
        if not self.n:
            return
        self.vb.setRange(xRange=(self.px.min(), self.px.max()),
                         yRange=(self.py.min(), self.py.max()), padding=0.04)

    def rebuild(self):
        p, q, limit = self.sp_p.value(), self.sp_q.value(), self.sp_n.value()
        t0 = time.perf_counter()
        t1 = t0
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            values, nxt = build_graph(p, q, limit, self.chk_rand.isChecked())
            nxt_l = nxt.tolist()
            cycles = find_cycles(nxt_l)
            pos, depth, sizes = layout(nxt, cycles)
            seg, head = edge_geometry(pos, nxt)
            t1 = time.perf_counter()

            self.p, self.q, self.n = p, q, len(values)
            self.values, self.nxt, self.depth = values, nxt_l, depth
            self.index = {v: i for i, v in enumerate(values)}
            self.indeg = np.bincount(nxt, minlength=self.n)
            self.px, self.py = pos[:, 0].copy(), pos[:, 1].copy()

            self.edges.setData(seg[:, 0], seg[:, 1], connect="pairs")
            self.heads.setData(head[:, 0], head[:, 1], connect="pairs")

            for it in self.node_items:
                self.vb.removeItem(it)
            self.node_items = []
            maxd = int(depth.max())
            pal = make_palette(maxd)
            base = 4 if self.n > 6000 else (5 if self.n > 2000 else 7)
            for d in range(maxd, -1, -1):
                m = depth == d
                it = pg.ScatterPlotItem(x=pos[m, 0], y=pos[m, 1],
                                        size=base + (5 if d == 0 else 0),
                                        brush=pal[d], pen=pg.mkPen(None), pxMode=True)
                it.setZValue(3)
                self.vb.addItem(it)
                self.node_items.append(it)

            for k, t in enumerate(self.label_pool):
                t.setVisible(False)
                self.label_idx[k] = -1
            self.select(None)
            self.set_legend(pal, maxd)
            self.set_summary(cycles, sizes, limit, t1 - t0)
            self.fit()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(
            f"Построено за {(t1 - t0) * 1000:.0f} мс: {self.n} вершин, "
            f"{int((seg.shape[0]) // 2)} рёбер")

    def set_legend(self, pal, maxd):
        parts = [f'<span style="color:{pal[d].color().name()}">●</span>&nbsp;{d}'
                 for d in range(min(maxd, 24) + 1)]
        if maxd > 24:
            parts.append("…")
        self.lb_legend.setText(
            "Цвет - число шагов до цикла (0 = сам цикл):<br>" + " &nbsp; ".join(parts))

    def set_summary(self, cycles, sizes, limit, secs):
        total = self.p ** self.q
        lines = [f"p = {self.p}, q = {self.q}, всего чисел: {total}",
                 f"Вершин в графе: {self.n} (лимит {limit})",
                 f"Компонент связности: {len(cycles)}", ""]
        order = sorted(range(len(cycles)), key=lambda i: -sizes[i])
        for i in order[:8]:
            c = cycles[i]
            s = " → ".join(self.lab(v) for v in c[:6]) + (" → …" if len(c) > 6 else "")
            kind = "неподвижная точка" if len(c) == 1 else f"цикл длины {len(c)}"
            lines.append(f"• {s}\n   {kind}, {sizes[i]} чисел")
        if len(order) > 8:
            lines.append(f"… и ещё {len(order) - 8}")
        self.txt.setPlainText("\n".join(lines))

    def nearest(self, scene_pos, tol_px=10):
        if not self.n or not self.vb.sceneBoundingRect().contains(scene_pos):
            return None
        pt = self.vb.mapSceneToView(scene_pos)
        sx, sy = self.vb.viewPixelSize()
        d2 = ((self.px - pt.x()) / sx) ** 2 + ((self.py - pt.y()) / sy) ** 2
        i = int(np.argmin(d2))
        return i if d2[i] <= tol_px ** 2 else None

    def on_move(self, args):
        i = self.nearest(args[0])
        if i is not None:
            self.statusBar().showMessage(
                f"{self.lab(i)} → {self.lab(self.nxt[i])}    "
                f"шагов до цикла: {self.depth[i]}")

    def on_click(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.select(self.nearest(ev.scenePos()))

    def on_search(self):
        s = self.ed_search.text().strip()
        if not s or not self.n:
            return
        try:
            v = int(s, self.p)
        except ValueError:
            self.statusBar().showMessage(
                f"«{s}» - не запись числа в системе с основанием {self.p}")
            return
        i = self.index.get(v)
        if i is None:
            self.statusBar().showMessage(
                "Этого числа нет в графе (не попало в выборку или все цифры равны)")
            return
        self.select(i)
        x, y = float(self.px[i]), float(self.py[i])
        self.vb.setRange(xRange=(x - 6, x + 6), yRange=(y - 6, y + 6), padding=0)

    def select(self, i):
        if i is None:
            self.hl_line.setData([], [])
            self.hl_nodes.setData(x=[], y=[])
            return
        path, seen, x = [i], {i}, self.nxt[i]
        while x not in seen:
            path.append(x)
            seen.add(x)
            x = self.nxt[x]
        ext = path + [x]
        self.hl_line.setData(self.px[ext], self.py[ext])
        self.hl_nodes.setData(x=self.px[path], y=self.py[path])
        cyc = path[path.index(x):]
        lines = [f"Число: {self.lab(i)}   (в десятичной: {self.values[i]})",
                 f"f(n) = {self.lab(self.nxt[i])}",
                 f"Прообразов в графе: {self.indeg[i]}",
                 f"Шагов до цикла: {self.depth[i]}",
                 "", "Путь:", " → ".join(self.lab(k) for k in ext), "",
                 ("Неподвижная точка" if len(cyc) == 1 else f"Цикл длины {len(cyc)}")
                 + ": " + " → ".join(self.lab(k) for k in cyc[:8])
                 + (" → …" if len(cyc) > 8 else "")]
        self.txt.setPlainText("\n".join(lines))

    def update_labels(self):
        if not self.n:
            return
        (x0, x1), (y0, y1) = self.vb.viewRange()
        vis = np.flatnonzero((self.px >= x0) & (self.px <= x1) &
                             (self.py >= y0) & (self.py <= y1))
        if len(vis) > LABEL_MAX:
            vis = vis[:0]
        while len(self.label_pool) < len(vis):
            t = pg.TextItem(color=(225, 230, 240), anchor=(0.5, 1.5),
                            fill=pg.mkBrush(18, 20, 26, 170))
            t.setZValue(10)
            t.setVisible(False)
            self.vb.addItem(t, ignoreBounds=True)
            self.label_pool.append(t)
            self.label_idx.append(-1)
        for k, t in enumerate(self.label_pool):
            i = int(vis[k]) if k < len(vis) else -1
            if self.label_idx[k] == i:
                continue
            self.label_idx[k] = i
            if i < 0:
                t.setVisible(False)
            else:
                t.setText(self.lab(i))
                t.setPos(float(self.px[i]), float(self.py[i]))
                t.setVisible(True)


def apply_dark_theme(app):
    app.setStyle("Fusion")
    R, C = QtGui.QPalette.ColorRole, QtGui.QColor
    pal = QtGui.QPalette()
    for role, col in ((R.Window, C(28, 30, 38)), (R.WindowText, C(215, 220, 230)),
                      (R.Base, C(20, 22, 28)), (R.AlternateBase, C(28, 30, 38)),
                      (R.Text, C(215, 220, 230)), (R.Button, C(44, 48, 60)),
                      (R.ButtonText, C(215, 220, 230)),
                      (R.Highlight, C(70, 120, 200)),
                      (R.HighlightedText, C(255, 255, 255)),
                      (R.ToolTipBase, C(44, 48, 60)),
                      (R.ToolTipText, C(215, 220, 230)),
                      (R.PlaceholderText, C(120, 128, 142))):
        pal.setColor(role, col)
    app.setPalette(pal)


def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_theme(app)
    win = KaprekarWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
