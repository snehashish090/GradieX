"""Matplotlib diagrams for GradieX networks.

Every function takes an optional ``ax`` (or ``fig``) and returns what it drew, so
the plots compose into larger figures instead of owning the screen. Nothing calls
``plt.show()`` unless you pass ``show=True``.

Colour
------
The palette is a validated set: three categorical hues whose worst all-pairs
colour-vision-deficiency separation is dE 9.2 (light) / 9.4 (dark), a single-hue
blue ramp for magnitude, and a blue-gray-red diverging ramp for signed values
such as weights. Identity is never carried by colour alone -- every chart with
more than one series ships a legend, and marker shape or a direct label repeats
the distinction. Light-mode aqua sits below 3:1 against the surface, so charts
that use it always label their marks.

``theme="dark"`` re-steps the same hues for a dark surface; it is a selected
palette, not an inversion.
"""

from __future__ import annotations

__all__ = [
    "plot_loss",
    "plot_architecture",
    "plot_weight_distribution",
    "plot_weight_heatmap",
    "plot_decision_boundary",
    "plot_predictions",
    "plot_loss_landscape",
    "summary",
    "THEMES",
]

# --------------------------------------------------------------------- palette

_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        # categorical slots 1-3, validated all-pairs on the light surface
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "sequential": _BLUE_RAMP,            # light -> dark: near-zero recedes
        "diverging": ("#2a78d6", "#f0efec", "#e34948"),
        "ring": "#fcfcfb",                   # surface-coloured ring on overlaps
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "sequential": list(reversed(_BLUE_RAMP)),  # dark -> light on a dark surface
        "diverging": ("#3987e5", "#383835", "#e66767"),
        "ring": "#1a1a19",
    },
}

_FONT_STACK = ["system-ui", "-apple-system", "Segoe UI", "Helvetica", "Arial",
               "DejaVu Sans", "sans-serif"]

# Secondary encoding: identity never rests on hue alone.
_MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*"]
_LINESTYLES = ["-", "--", "-.", ":"]


def _theme(name):
    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(
            f"unknown theme {name!r}; available: {', '.join(sorted(THEMES))}") from None


def _mpl():
    """Import matplotlib lazily.

    The engine itself has no Python dependencies, and importing pyplot costs a
    few hundred milliseconds, so nothing is imported until a plot is asked for.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "plotting needs matplotlib. Install it with `pip install matplotlib`, "
            "or `pip install gradiex` which pulls it in."
        ) from exc
    return matplotlib, plt


def _new_axes(ax, theme, figsize):
    """Return (fig, ax), creating a themed figure when the caller passed none."""
    _, plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    return fig, ax


def _style_axes(ax, theme, *, grid_axis="y"):
    """Recessive chrome: hairline grid, no top/right spines, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=3, width=1.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(theme["secondary"])
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=theme["grid"], linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)


def _title(ax, theme, title, subtitle=None):
    if title:
        ax.set_title(title, color=theme["primary"], fontsize=12, loc="left",
                     pad=22 if subtitle else 8, fontweight="medium")
    if subtitle:
        # Sits in the gap the title's pad opened up; any closer and the title's
        # descenders touch it.
        ax.text(0.0, 1.018, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                color=theme["muted"], fontsize=9)


def _legend(ax, theme, *, loc="best", over_data=False):
    """A legend whose text wears ink, never the series colour.

    ``over_data`` gives it a surface-coloured frame, so it stays readable when it
    lands on top of a filled region.
    """
    handles, _ = ax.get_legend_handles_labels()
    if not handles:
        return None
    legend = ax.legend(
        loc=loc, fontsize=9,
        frameon=over_data,
        facecolor=theme["surface"] if over_data else "none",
        edgecolor="none",
        framealpha=0.92 if over_data else 0.0,
    )
    for text in legend.get_texts():
        text.set_color(theme["secondary"])
    legend.set_zorder(6)
    return legend


def _apply_font():
    matplotlib, _ = _mpl()
    matplotlib.rcParams["font.family"] = "sans-serif"
    existing = matplotlib.rcParams["font.sans-serif"]
    wanted = [f for f in _FONT_STACK if f not in existing]
    matplotlib.rcParams["font.sans-serif"] = wanted + list(existing)


def _finish(fig, show):
    import warnings
    _, plt = _mpl()
    # A colorbar's axes is not tight_layout-compatible and matplotlib warns about
    # it on every heatmap. The layout it produces is the one we want, so the
    # warning is noise rather than a signal.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*not compatible with tight_layout.*",
            category=UserWarning)
        fig.tight_layout()
    if show:
        plt.show()
    return fig


def _diverging_cmap(theme):
    matplotlib, _ = _mpl()
    from matplotlib.colors import LinearSegmentedColormap
    low, mid, high = theme["diverging"]
    return LinearSegmentedColormap.from_list("gradiex_div", [low, mid, high], N=256)


def _sequential_cmap(theme):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("gradiex_seq", theme["sequential"], N=256)


def _moving_average(values, window):
    import numpy as np
    if window <= 1:
        return np.asarray(values, dtype=float)
    v = np.asarray(values, dtype=float)
    window = min(int(window), len(v))
    kernel = np.ones(window) / window
    # 'same' would taper the ends toward zero; pad with the edge values instead.
    pad = window // 2
    padded = np.pad(v, (pad, window - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ------------------------------------------------------------------ loss curve

def plot_loss(history, *, val_history=None, labels=("training", "validation"),
              smooth=None, log=False, ax=None, theme="light", title="Training loss",
              figsize=(7.0, 4.0), show=False):
    """Plot the per-epoch loss returned by :meth:`Network.train`.

    Parameters
    ----------
    history : sequence of float
        One loss per epoch.
    val_history : sequence of float, optional
        A second curve (for example a validation split) drawn alongside.
    smooth : int, optional
        Window for a moving-average overlay. The raw curve stays visible
        underneath in a lighter step of the same hue, because the smoothing is a
        reading aid for one series, not a second series.
    log : bool
        Log-scale the y axis. Useful once the loss spans orders of magnitude;
        only valid while every plotted value is > 0.
    """
    import numpy as np
    _apply_font()
    th = _theme(theme)
    y = np.asarray(list(history), dtype=float)
    if y.size == 0:
        raise ValueError("history is empty - train() returns [] for 0 epochs or an "
                         "empty dataset, and there is nothing to plot")

    fig, ax = _new_axes(ax, th, figsize)
    epochs = np.arange(1, y.size + 1)

    series = [(epochs, y, th["series"][0], labels[0], "-")]
    if val_history is not None:
        v = np.asarray(list(val_history), dtype=float)
        if v.size != y.size:
            raise ValueError(f"val_history has {v.size} entries but history has {y.size}")
        series.append((epochs, v, th["series"][1], labels[1], "--"))

    if log and min(float(s[1].min()) for s in series) <= 0.0:
        raise ValueError("log=True needs every plotted loss to be > 0")

    if smooth and val_history is None:
        # One series, two renderings: raw recedes, the smoothed line carries it.
        ax.plot(epochs, y, color=_BLUE_RAMP[3], linewidth=1.0, zorder=2,
                label="per epoch")
        ax.plot(epochs, _moving_average(y, smooth), color=th["series"][0],
                linewidth=2.0, zorder=3, label=f"{int(smooth)}-epoch mean")
    else:
        for x, values, color, label, style in series:
            ax.plot(x, values, color=color, linewidth=2.0, linestyle=style,
                    zorder=3, label=label)

    if log:
        ax.set_yscale("log")

    ax.set_xlabel("epoch", color=th["secondary"], fontsize=10)
    ax.set_ylabel("loss", color=th["secondary"], fontsize=10)
    ax.set_xlim(1, max(int(y.size), 2))
    _style_axes(ax, th)

    # Selective direct label: the final value is the number people look for.
    final = float(y[-1])
    ax.annotate(f"{final:.4g}", xy=(epochs[-1], final),
                xytext=(-4, 8), textcoords="offset points",
                ha="right", va="bottom", fontsize=9, color=th["primary"],
                fontweight="medium")

    drop = ""
    if y.size > 1 and y[0] > 0:
        drop = f"  ·  {y[0]:.4g} → {final:.4g}"
    _title(ax, th, title, f"{y.size} epochs{drop}")

    handles, _ = ax.get_legend_handles_labels()
    if len(handles) > 1:
        _legend(ax, th, loc="upper right")
    return _finish(fig, show)


# --------------------------------------------------------- architecture diagram

def plot_architecture(net, *, max_nodes=8, ax=None, theme="light",
                      title="Network architecture", figsize=(8.0, 4.5), show=False):
    """Draw the layer graph: one column per layer, labelled with width and activation.

    Layers wider than ``max_nodes`` are drawn truncated with an ellipsis; the
    label always states the true width, so the diagram never misreports the
    shape it is compressing.
    """
    import numpy as np
    _apply_font()
    th = _theme(theme)
    specs = list(net.layers)
    if not specs:
        raise ValueError("the network has no layers yet")

    fig, ax = _new_axes(ax, th, figsize)

    # Columns: the input vector, then every declared layer.
    columns = [{"kind": "input", "width": specs[0]["input_size"], "label": "input",
                "sub": f"{specs[0]['input_size']} features"}]
    for spec in specs:
        columns.append({
            "kind": spec["kind"],
            "width": spec["width"],
            "label": "output" if spec["kind"] == "output" else "hidden",
            "sub": f"{spec['width']} × {spec['activation']}",
            "loss": spec.get("loss"),
        })

    colour_of = {"input": th["series"][2], "hidden": th["series"][0],
                 "output": th["series"][1]}

    # Node positions, vertically centred per column.
    positions = []
    for i, col in enumerate(columns):
        shown = min(col["width"], max_nodes)
        ys = np.linspace(-(shown - 1) / 2.0, (shown - 1) / 2.0, shown) if shown > 1 else np.array([0.0])
        positions.append([(float(i), float(y)) for y in ys])

    # Edges first so nodes sit on top of them.
    for left, right in zip(positions, positions[1:]):
        for x0, y0 in left:
            for x1, y1 in right:
                ax.plot([x0, x1], [y0, y1], color=th["grid"], linewidth=0.7,
                        zorder=1, solid_capstyle="round")

    for col, pts in zip(columns, positions):
        colour = colour_of[col["kind"]]
        for x, y in pts:
            ax.scatter([x], [y], s=190, color=colour, zorder=3,
                       edgecolors=th["ring"], linewidths=2.0)
        if col["width"] > len(pts):
            ax.text(pts[0][0], min(p[1] for p in pts) - 0.75, "⋮",
                    ha="center", va="center", color=th["muted"], fontsize=14, zorder=3)

    top = max(max(p[1] for p in pts) for pts in positions)
    bottom = min(min(p[1] for p in pts) for pts in positions)

    # Direct labels under every column: this is what carries identity, so the
    # sub-3:1 light-mode aqua never has to.
    for col, pts in zip(columns, positions):
        x = pts[0][0]
        ax.text(x, bottom - 1.6, col["label"], ha="center", va="top",
                color=th["primary"], fontsize=10, fontweight="medium")
        ax.text(x, bottom - 2.1, col["sub"], ha="center", va="top",
                color=th["muted"], fontsize=9)

    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.set_ylim(bottom - 3.0, top + 0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    out = specs[-1]
    params = sum(s["width"] * s["input_size"] + s["width"] for s in specs)
    subtitle = f"{params:,} parameters"
    if out.get("loss"):
        subtitle += f"  ·  loss {out['loss']}"
    if getattr(net, "built", False):
        subtitle += f"  ·  lr {net.learning_rate:g}  ·  batch {net.batch_size}"
    _title(ax, th, title, subtitle)
    return _finish(fig, show)


# ----------------------------------------------------- weight distributions

def plot_weight_distribution(net, *, bins=40, fig=None, theme="light",
                             title="Weight distribution by layer",
                             figsize=None, show=False):
    """Small multiples: one histogram of the weight values per layer.

    Small multiples rather than overlaid histograms, so no layer's colour has to
    be told apart from another's -- the facet title carries identity. Watch for a
    layer collapsing toward zero (a dead layer) or spreading far wider than the
    others (the one about to diverge).
    """
    import numpy as np
    _apply_font()
    th = _theme(theme)
    _, plt = _mpl()
    if not net.built:
        raise RuntimeError("call build() before inspecting weights")

    specs = list(net.layers)
    n = len(specs)
    if figsize is None:
        figsize = (min(3.1 * n, 13.0), 3.2)
    if fig is None:
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
        axes = axes[0]
    else:
        axes = fig.subplots(1, n, squeeze=False)[0]
    fig.patch.set_facecolor(th["surface"])

    for i, (ax, spec) in enumerate(zip(axes, specs)):
        w = np.asarray(net.weights(i)).ravel()
        ax.set_facecolor(th["surface"])
        ax.hist(w, bins=bins, color=th["series"][0], edgecolor=th["surface"],
                linewidth=0.5, zorder=3)
        ax.axvline(0.0, color=th["axis"], linewidth=1.0, zorder=2)
        _style_axes(ax, th)
        name = "output" if spec["kind"] == "output" else f"hidden {i}"
        ax.set_title(f"{name}  ({spec['width']}×{spec['input_size']})",
                     color=th["primary"], fontsize=10, loc="left", pad=6)
        ax.text(0.0, 1.02, f"σ {w.std():.3g}  ·  |w|max {np.abs(w).max():.3g}",
                transform=ax.transAxes, ha="left", va="bottom",
                color=th["muted"], fontsize=8)
        ax.set_xlabel("weight", color=th["secondary"], fontsize=9)
        if i == 0:
            ax.set_ylabel("count", color=th["secondary"], fontsize=9)

    fig.suptitle(title, color=th["primary"], fontsize=12, x=0.01, ha="left",
                 fontweight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if show:
        plt.show()
    return fig


def plot_weight_heatmap(net, layer=-1, *, ax=None, theme="light", title=None,
                        figsize=(6.0, 4.2), show=False):
    """Heatmap of one layer's weight matrix, rows = neurons, columns = inputs.

    Weights are signed, so the ramp is diverging (blue negative, gray zero, red
    positive) and the scale is symmetric about zero -- a sequential ramp here
    would hide the sign, which is the thing worth seeing.
    """
    import numpy as np
    _apply_font()
    th = _theme(theme)
    if not net.built:
        raise RuntimeError("call build() before inspecting weights")

    w = np.asarray(net.weights(layer))
    fig, ax = _new_axes(ax, th, figsize)

    limit = float(np.abs(w).max()) or 1.0
    image = ax.imshow(w, cmap=_diverging_cmap(th), vmin=-limit, vmax=limit,
                      aspect="auto", interpolation="nearest")

    ax.set_xlabel("input", color=th["secondary"], fontsize=10)
    ax.set_ylabel("neuron", color=th["secondary"], fontsize=10)
    # Both axes index discrete units; fractional ticks would be meaningless.
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    _style_axes(ax, th, grid_axis=None)

    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=th["muted"], labelsize=8)
    bar.set_label("weight", color=th["secondary"], fontsize=9)

    specs = list(net.layers)
    index = layer if layer >= 0 else len(specs) + layer
    spec = specs[index]
    name = "output" if spec["kind"] == "output" else f"hidden {index}"
    _title(ax, th, title or f"{name} weights",
           f"{w.shape[0]} neurons × {w.shape[1]} inputs  ·  symmetric about 0, ±{limit:.3g}")
    return _finish(fig, show)


# ------------------------------------------------------- decision boundary

def plot_decision_boundary(net, X, y=None, *, resolution=220, ax=None,
                           theme="light", title="Decision boundary",
                           figsize=(6.0, 5.2), show=False):
    """Shade what the network predicts across a 2-D input plane.

    Only defined for a network with ``input_size == 2``; that covers the toy
    problems (XOR and friends) this is most useful on.

    A single output is read as a probability and shaded with the diverging ramp
    about 0.5. Several outputs are read as classes via ``argmax`` and shaded with
    lightened categorical hues; the sample markers repeat the class in shape as
    well as colour.
    """
    import numpy as np
    _apply_font()
    th = _theme(theme)
    if not net.built:
        raise RuntimeError("call build() before plotting a decision boundary")
    if net.input_size != 2:
        raise ValueError(
            f"a decision boundary needs a 2-feature input; this network takes "
            f"{net.input_size}. Project the data to 2-D first, or use plot_predictions().")

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must have shape (n_samples, 2), got {X.shape}")

    fig, ax = _new_axes(ax, th, figsize)

    # np.ptp, not ndarray.ptp: the method was removed in NumPy 2.
    pad_x = 0.1 * (float(np.ptp(X[:, 0])) or 1.0)
    pad_y = 0.1 * (float(np.ptp(X[:, 1])) or 1.0)
    xs = np.linspace(X[:, 0].min() - pad_x, X[:, 0].max() + pad_x, resolution)
    ys = np.linspace(X[:, 1].min() - pad_y, X[:, 1].max() + pad_y, resolution)
    gx, gy = np.meshgrid(xs, ys)

    grid = np.column_stack([gx.ravel(), gy.ravel()])
    preds = np.array([net.predict(point) for point in grid])
    n_out = preds.shape[1]

    if n_out == 1:
        z = preds[:, 0].reshape(gx.shape)
        mesh = ax.pcolormesh(gx, gy, z, cmap=_diverging_cmap(th), vmin=0.0, vmax=1.0,
                             shading="auto", zorder=1)
        ax.contour(gx, gy, z, levels=[0.5], colors=[th["surface"]],
                   linewidths=2.2, linestyles="-", zorder=2)
        ax.contour(gx, gy, z, levels=[0.5], colors=[th["primary"]],
                   linewidths=1.2, linestyles="--", zorder=3)
        bar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.03)
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors=th["muted"], labelsize=8)
        bar.set_label("output", color=th["secondary"], fontsize=9)
        subtitle = "dashed line: the 0.5 contour"
    else:
        from matplotlib.colors import ListedColormap
        classes = preds.argmax(axis=1).reshape(gx.shape)
        shown = min(n_out, len(th["series"]))
        faded = [_blend(th["series"][i % len(th["series"])], th["surface"], 0.78)
                 for i in range(n_out)]
        ax.pcolormesh(gx, gy, classes, cmap=ListedColormap(faded),
                      vmin=-0.5, vmax=n_out - 0.5, shading="auto", zorder=1)
        subtitle = f"{n_out} classes by argmax"
        if n_out > shown:
            subtitle += f"  ·  hues repeat past {shown}"

    if y is not None:
        y = np.asarray(y)
        classes = y.argmax(axis=1) if y.ndim == 2 and y.shape[1] > 1 else np.asarray(
            y, dtype=float).ravel().round().astype(int)
        # Over a diverging probability field, a blue or red marker would sit on
        # its own colour and vanish, so the binary case puts the samples in ink
        # and lets marker shape carry the class. The multiclass regions are
        # faded far enough that the full hue still reads on top of them.
        for k in sorted(set(classes.tolist())):
            sel = classes == k
            face = th["primary"] if n_out == 1 else th["series"][k % len(th["series"])]
            ax.scatter(X[sel, 0], X[sel, 1], color=face,
                       marker=_MARKERS[k % len(_MARKERS)], s=90,
                       edgecolors=th["ring"], linewidths=1.8, zorder=4,
                       label=f"class {k}")
        _legend(ax, th, loc="best", over_data=True)

    ax.set_xlabel("feature 0", color=th["secondary"], fontsize=10)
    ax.set_ylabel("feature 1", color=th["secondary"], fontsize=10)
    _style_axes(ax, th, grid_axis=None)
    _title(ax, th, title, subtitle)
    return _finish(fig, show)


def _blend(colour, towards, amount):
    """Mix ``colour`` toward ``towards`` by ``amount`` in [0, 1]."""
    from matplotlib.colors import to_rgb
    a = to_rgb(colour)
    b = to_rgb(towards)
    return tuple(a[i] * (1 - amount) + b[i] * amount for i in range(3))


# ------------------------------------------------------------- predictions

def plot_predictions(net, X, Y, *, kind="auto", ax=None, theme="light", title=None,
                     figsize=(5.6, 5.0), show=False):
    """Compare predictions against targets.

    ``kind="regression"`` draws predicted against actual with the y = x reference
    line; ``kind="classification"`` draws a confusion matrix. ``"auto"`` picks
    the confusion matrix when the network has several outputs or the targets are
    all 0/1, and the scatter otherwise.
    """
    import numpy as np
    _apply_font()
    th = _theme(theme)
    if not net.built:
        raise RuntimeError("call build() before plotting predictions")

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got {X.ndim}-D")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D (n_samples, n_targets), got {Y.ndim}-D")
    if len(X) != len(Y):
        raise ValueError(f"X has {len(X)} rows but Y has {len(Y)}")

    preds = np.array([net.predict(row) for row in X])

    if kind == "auto":
        binary = np.all((Y == 0) | (Y == 1))
        kind = "classification" if (Y.shape[1] > 1 or binary) else "regression"
    if kind not in ("regression", "classification"):
        raise ValueError(f"kind must be 'auto', 'regression' or 'classification', got {kind!r}")

    fig, ax = _new_axes(ax, th, figsize)

    if kind == "regression":
        actual = Y.ravel()
        predicted = preds.ravel()
        lo = float(min(actual.min(), predicted.min()))
        hi = float(max(actual.max(), predicted.max()))
        span = (hi - lo) or 1.0
        lo, hi = lo - 0.05 * span, hi + 0.05 * span

        ax.plot([lo, hi], [lo, hi], color=th["axis"], linewidth=1.5,
                linestyle="--", zorder=2, label="perfect")
        ax.scatter(actual, predicted, s=55, color=th["series"][0],
                   edgecolors=th["ring"], linewidths=1.5, zorder=3, label="prediction")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("actual", color=th["secondary"], fontsize=10)
        ax.set_ylabel("predicted", color=th["secondary"], fontsize=10)
        _style_axes(ax, th, grid_axis="both")

        residual = predicted - actual
        denom = float(((actual - actual.mean()) ** 2).sum())
        r2 = 1.0 - float((residual ** 2).sum()) / denom if denom > 0 else float("nan")
        subtitle = f"{len(actual)} points  ·  RMSE {np.sqrt((residual ** 2).mean()):.4g}"
        if denom > 0:
            subtitle += f"  ·  R² {r2:.3f}"
        _legend(ax, th, loc="upper left", over_data=True)
        _title(ax, th, title or "Predicted vs actual", subtitle)
        return _finish(fig, show)

    # classification: confusion matrix, magnitude -> single-hue ramp
    if Y.shape[1] > 1:
        true = Y.argmax(axis=1)
        guess = preds.argmax(axis=1)
        n_classes = Y.shape[1]
    else:
        true = Y.ravel().round().astype(int)
        guess = (preds.ravel() >= 0.5).astype(int)
        n_classes = 2

    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, g in zip(true, guess):
        matrix[t, g] += 1

    image = ax.imshow(matrix, cmap=_sequential_cmap(th), aspect="equal",
                      interpolation="nearest", vmin=0)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xlabel("predicted", color=th["secondary"], fontsize=10)
    ax.set_ylabel("actual", color=th["secondary"], fontsize=10)
    _style_axes(ax, th, grid_axis=None)

    # Every cell is labelled, so the counts never depend on reading the ramp.
    peak = matrix.max() or 1
    for i in range(n_classes):
        for j in range(n_classes):
            value = matrix[i, j]
            dark_cell = (value / peak) > 0.55
            if theme == "dark":
                dark_cell = not dark_cell
            ax.text(j, i, str(value), ha="center", va="center", fontsize=10,
                    fontweight="medium",
                    color=th["surface"] if dark_cell else th["primary"])

    accuracy = float((true == guess).mean())
    _title(ax, th, title or "Confusion matrix",
           f"{len(true)} samples  ·  accuracy {accuracy:.1%}")
    return _finish(fig, show)


# ----------------------------------------------------------- loss landscape

def _filter_normalised_direction(weights, rng):
    """A random direction scaled, per neuron, to that neuron's own weight norm.

    This normalisation is what makes a loss landscape mean anything. A network's
    loss is invariant to rescaling a neuron's weights (a neuron with tiny weights
    and one with large weights can compute the same function), so an unnormalised
    random step moves the small-weight neurons a long way in function space and
    the large-weight ones barely at all. The resulting picture says more about
    the weight scales than about the loss. Scaling each direction row to the norm
    of the row it perturbs removes that, and is the filter normalisation from
    Li et al., "Visualizing the Loss Landscape of Neural Nets" (2018).
    """
    import numpy as np
    direction = rng.normal(size=weights.shape).astype(np.float32)
    # Row = one neuron's incoming weights.
    weight_norm = np.linalg.norm(weights, axis=1, keepdims=True)
    direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    # A neuron with all-zero weights has no scale to match, so it stays put
    # rather than being handed an arbitrary one.
    scale = np.divide(weight_norm, direction_norm,
                      out=np.zeros_like(weight_norm),
                      where=direction_norm > 0)
    return direction * scale


def plot_loss_landscape(net, X, Y, *, span=1.0, resolution=25, seed=None,
                        log=False, contours=True, elev=34.0, azim=-58.0,
                        ax=None, theme="light", title="Loss landscape",
                        figsize=(7.6, 6.2), show=False):
    """Plot the loss surface on a 2-D slice through parameter space.

    Two filter-normalised random directions are drawn through the trained
    weights, and the loss over ``(X, Y)`` is measured on a grid of steps along
    them. ``(0, 0)`` is the trained model; the surface around it is the basin it
    settled in -- broad and smooth, or narrow and ragged.

    This is a **random 2-D slice of a very high-dimensional surface**, so read it
    as a texture rather than a map: it shows what the neighbourhood is like, not
    where the optimiser went or where other minima are. A different ``seed``
    gives a different slice.

    The trained model is always at ``(0, 0)`` by construction, and the subtitle
    says so, because matplotlib's 3-D renderer has no depth buffer: the marker
    drawn there is best-effort and the surface may pass in front of it. Rotate
    with ``elev`` / ``azim`` if it matters for a particular figure.

    Parameters
    ----------
    span : float
        How far to step along each direction, in units of the filter-normalised
        direction. 1.0 means "a step the size of each neuron's own weights".
    resolution : int
        Grid points per axis. Cost is ``resolution**2`` passes over the dataset,
        each one engine-side, so 25 is a good default and 60 is still quick on a
        small dataset.
    seed : int, optional
        Seeds the two directions, so a landscape can be reproduced.
    log : bool
        Plot log10 of the loss. Landscapes often span orders of magnitude around
        a good minimum, where a linear axis shows a flat plain and one spike.
    contours : bool
        Project filled contours onto the floor of the box.
    """
    import numpy as np
    _apply_font()
    th = _theme(theme)
    _, plt = _mpl()
    if not net.built:
        raise RuntimeError("call build() before plotting a loss landscape")
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    if not span > 0:
        raise ValueError(f"span must be > 0, got {span}")

    n_layers = len(list(net.layers))
    rng = np.random.default_rng(seed)

    # Snapshot first: the sweep walks the live network away from its trained
    # point, and it has to land back exactly where it started.
    base = [np.asarray(net.weights(i), dtype=np.float32).copy() for i in range(n_layers)]
    first = [_filter_normalised_direction(w, rng) for w in base]
    second = [_filter_normalised_direction(w, rng) for w in base]

    steps = np.linspace(-span, span, resolution)
    grid_x, grid_y = np.meshgrid(steps, steps)
    surface = np.empty_like(grid_x, dtype=float)

    try:
        for row in range(resolution):
            for col in range(resolution):
                a = grid_x[row, col]
                b = grid_y[row, col]
                for i in range(n_layers):
                    net.set_weights(i, base[i] + a * first[i] + b * second[i])
                surface[row, col] = net.evaluate(X, Y)
    finally:
        # Including on an exception: a half-swept network would look trained.
        for i in range(n_layers):
            net.set_weights(i, base[i])

    centre = float(net.evaluate(X, Y))

    finite = np.isfinite(surface)
    if not finite.any():
        raise ValueError("every point on the landscape was non-finite; try a smaller span")
    surface = np.where(finite, surface, np.nan)

    label = "loss"
    centre_height = centre
    if log:
        lowest = np.nanmin(surface)
        if lowest <= 0:
            raise ValueError("log=True needs every sampled loss to be > 0; "
                             f"the smallest was {lowest:.6g}")
        surface = np.log10(surface)
        centre_height = float(np.log10(centre))
        label = "log₁₀ loss"

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
    else:
        if not hasattr(ax, "plot_surface"):
            raise ValueError("ax must be a 3-D axes, e.g. "
                             "fig.add_subplot(projection='3d')")
        fig = ax.figure
    fig.patch.set_facecolor(th["surface"])
    ax.set_facecolor(th["surface"])

    cmap = _sequential_cmap(th)
    # Slightly translucent: matplotlib's 3-D renderer has no depth buffer, and an
    # opaque surface hides the floor contours and the trained-point marker
    # whenever it happens to arc over them.
    ax.plot_surface(grid_x, grid_y, surface, cmap=cmap, linewidth=0.0,
                    antialiased=True, rstride=1, cstride=1, alpha=0.9)

    peak = float(np.nanmax(surface))
    floor = float(np.nanmin(surface))
    depth = peak - floor
    if contours:
        floor -= 0.18 * (depth or 1.0)
        ax.contourf(grid_x, grid_y, surface, zdir="z", offset=floor, cmap=cmap,
                    levels=14, alpha=0.75)
    ax.set_zlim(floor, peak)

    # The trained model sits at (0, 0). A marker on the surface there would be
    # occluded by the surface itself -- matplotlib's 3-D renderer has no true
    # depth buffer -- so it is marked on the floor, where it is always visible,
    # with a dropline up to the height it actually reached.
    ax.plot([0.0, 0.0], [0.0, 0.0], [floor, centre_height], color=th["series"][1],
            linewidth=1.2, linestyle="--", zorder=5)
    ax.scatter([0.0], [0.0], [floor], color=th["series"][1], s=80, marker="o",
               edgecolors=th["ring"], linewidths=1.5, depthshade=False, zorder=6,
               label="trained weights")
    ax.scatter([0.0], [0.0], [centre_height], color=th["series"][1], s=45, marker="o",
               edgecolors=th["ring"], linewidths=1.2, depthshade=False, zorder=7)

    ax.set_xlabel("direction 1", color=th["secondary"], fontsize=10, labelpad=8)
    ax.set_ylabel("direction 2", color=th["secondary"], fontsize=10, labelpad=8)
    # Height already encodes the loss and the floor contours share its ramp, so
    # a colorbar would key the same number a third time.
    ax.set_zlabel(label, color=th["secondary"], fontsize=10, labelpad=8)
    ax.view_init(elev=elev, azim=azim)
    _style_3d(ax, th)

    _legend(ax, th, loc="upper right")
    # A 3-D axes puts its title above the box rather than above the axes area,
    # so the header is placed on the figure instead and stacked by hand.
    fig.text(0.03, 0.97, title, ha="left", va="top", color=th["primary"],
             fontsize=12, fontweight="medium")
    fig.text(0.03, 0.925,
             f"filter-normalised random slice  ·  {resolution}×{resolution} grid  ·  "
             f"(0, 0) is the trained model, loss {centre:.4g}",
             ha="left", va="top", color=th["muted"], fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.90, bottom=0.06)
    if show:
        plt.show()
    return fig


def _style_3d(ax, theme):
    """Recessive chrome for a 3-D axes: surface-coloured panes, hairline grid."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_pane_color(theme["surface"])
        except AttributeError:            # older matplotlib
            axis.pane.set_facecolor(theme["surface"])
        axis.pane.set_edgecolor(theme["grid"])
        axis.pane.set_alpha(1.0)
        axis.line.set_color(theme["axis"])
        axis._axinfo["grid"]["color"] = theme["grid"]
        axis._axinfo["grid"]["linewidth"] = 0.8
    ax.tick_params(colors=theme["muted"], labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        label.set_color(theme["secondary"])


# ---------------------------------------------------------------- dashboard

def summary(net, history=None, X=None, Y=None, *, theme="light",
            title="Network summary", figsize=(12.0, 8.0), show=False):
    """One figure: architecture, loss, weight distribution, and -- when the data
    allows it -- the decision boundary or a predicted-vs-actual panel.

    Every argument past ``net`` is optional; panels with nothing to draw are
    left out rather than shown empty.
    """
    _apply_font()
    th = _theme(theme)
    _, plt = _mpl()

    panels = ["architecture"]
    if history is not None and len(history):
        panels.append("loss")
    panels.append("weights")
    can_shade = X is not None and net.built and net.input_size == 2
    if can_shade:
        panels.append("boundary")
    elif X is not None and Y is not None:
        panels.append("predictions")

    columns = 2
    rows = (len(panels) + columns - 1) // columns
    fig = plt.figure(figsize=(figsize[0], figsize[1] * rows / 2.0))
    fig.patch.set_facecolor(th["surface"])
    grid = fig.add_gridspec(rows, columns, hspace=0.45, wspace=0.28)

    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        if panel == "weights":
            # Nested small multiples need their own sub-grid.
            layer_count = len(list(net.layers))
            sub = grid[row, column].subgridspec(1, layer_count, wspace=0.35)
            import numpy as np
            for i, spec in enumerate(net.layers):
                ax = fig.add_subplot(sub[0, i])
                w = np.asarray(net.weights(i)).ravel()
                ax.set_facecolor(th["surface"])
                ax.hist(w, bins=30, color=th["series"][0], edgecolor=th["surface"],
                        linewidth=0.5, zorder=3)
                ax.axvline(0.0, color=th["axis"], linewidth=1.0, zorder=2)
                _style_axes(ax, th)
                name = "output" if spec["kind"] == "output" else f"hidden {i}"
                ax.set_title(f"{name}  σ {w.std():.3g}", color=th["primary"],
                             fontsize=9, loc="left", pad=4)
            continue

        ax = fig.add_subplot(grid[row, column])
        if panel == "architecture":
            plot_architecture(net, ax=ax, theme=theme)
        elif panel == "loss":
            plot_loss(history, ax=ax, theme=theme)
        elif panel == "boundary":
            plot_decision_boundary(net, X, Y, ax=ax, theme=theme)
        elif panel == "predictions":
            plot_predictions(net, X, Y, ax=ax, theme=theme)

    fig.suptitle(title, color=th["primary"], fontsize=14, x=0.01, ha="left",
                 fontweight="medium")
    if show:
        plt.show()
    return fig
