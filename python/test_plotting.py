# Headless checks for gradiex.plotting.
#
# These assert that every diagram builds a figure with the axes and artists it
# claims, and that the guards raise instead of drawing something meaningless.
# They do not assert on pixels -- rendering is checked by eye.

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")          # no display needed, and no window is opened
import matplotlib.pyplot as plt
import numpy as np

import gradiex

XOR_X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
XOR_Y = np.array([[0.], [1.], [1.], [0.]])

failures = []


# ------------------------------------------------------------------ harness

def _assert(condition, message="assertion failed"):
    if not condition:
        raise AssertionError(message)


def check(label, fn):
    try:
        fn()
        print(f"  ok  {label}")
    except Exception as exc:                      # noqa: BLE001 - reporting harness
        failures.append((label, exc))
        print(f"  FAIL {label}: {type(exc).__name__}: {exc}")


def expect_raises(label, fn, exc_type=Exception):
    try:
        fn()
    except exc_type as exc:
        print(f"  ok  {label}: {type(exc).__name__}: {str(exc)[:58]}")
        return
    except Exception as exc:                      # noqa: BLE001
        failures.append((label, exc))
        print(f"  FAIL {label}: raised {type(exc).__name__}, wanted {exc_type.__name__}")
        return
    failures.append((label, AssertionError("no error")))
    print(f"  FAIL {label}: no error raised")


def one_axes(fig, expected=1, artists=None):
    _assert(fig is not None, "no figure")
    _assert(len(fig.axes) >= expected,
            f"expected >= {expected} axes, got {len(fig.axes)}")
    if artists:
        artists(fig)
    plt.close(fig)


# ------------------------------------------------------------------ fixtures

def build_xor(epochs=400):
    gradiex.seed(42)
    net = gradiex.Network(learning_rate=0.5, batch_size=4)
    net.add_hidden(6, input_size=2, activation="tanh", init="xavier")
    net.add_output(1, activation="sigmoid", loss="mse", init="xavier")
    net.build()
    return net, net.train(XOR_X, XOR_Y, epochs=epochs, shuffle=True, seed=42)


def wide_net():
    gradiex.seed(1)
    m = gradiex.Network(0.1)
    m.add_hidden(32, input_size=5, activation="relu")
    m.add_output(3, activation="sigmoid", loss="mse")
    m.build()
    return m


def untrained_net():
    gradiex.seed(1)
    m = gradiex.Network(0.1)
    m.add_hidden(2, input_size=2)
    m.add_output(1)
    m.build()
    return m


def unbuilt_net():
    m = gradiex.Network(0.1)
    m.add_hidden(2, input_size=2)
    m.add_output(1)
    return m


# ------------------------------------------------------------------- checks

print("gradiex", gradiex.__version__, "| matplotlib", matplotlib.__version__)

print("\npackage layout:")
check("Network subclasses the extension type",
      lambda: _assert(issubclass(gradiex.Network, gradiex._core.Network)))
check("engine symbols re-exported",
      lambda: _assert(callable(gradiex.seed) and len(gradiex.activations()) > 1))
check("plotting functions re-exported",
      lambda: _assert(all(hasattr(gradiex, n) for n in
                          ("plot_loss", "plot_architecture", "summary"))))

net, history = build_xor()

print("\ntrain(plot=...):")


def train_plots_and_remembers():
    gradiex.seed(7)
    m = gradiex.Network(0.5, batch_size=4)
    m.add_hidden(4, input_size=2, activation="tanh", init="xavier")
    m.add_output(1, activation="sigmoid", loss="mse", init="xavier")
    m.build()
    h = m.train(XOR_X, XOR_Y, epochs=50, plot=True)
    _assert(m.history == h, "history not remembered")
    _assert(m.last_figure is not None, "no figure produced")
    _assert(len(m.last_figure.axes) == 1, "expected one axes")
    plt.close(m.last_figure)


check("plot=True draws and stores the figure", train_plots_and_remembers)
check("history is remembered without plot=True",
      lambda: _assert(net.history == history and len(history) == 400))
check("training still matches the engine (loss fell)",
      lambda: _assert(history[-1] < history[0] / 10, "did not learn"))
expect_raises("plot kwargs without plot=True",
              lambda: net.train(XOR_X, XOR_Y, epochs=1, smooth=5), TypeError)

print("\nfigures:")
check("plot_loss", lambda: one_axes(
    net.plot_loss(), artists=lambda f: _assert(len(f.axes[0].lines) == 1, "one line")))
check("plot_loss(smooth=) draws raw + mean and a legend", lambda: one_axes(
    net.plot_loss(smooth=20),
    artists=lambda f: _assert(len(f.axes[0].lines) == 2 and f.axes[0].get_legend(),
                              "two lines and a legend")))
check("plot_loss(val_history=)", lambda: one_axes(
    gradiex.plot_loss(history, val_history=[v * 1.1 for v in history]),
    artists=lambda f: _assert(len(f.axes[0].lines) == 2, "two lines")))
check("plot_loss(log=True)", lambda: one_axes(
    net.plot_loss(log=True),
    artists=lambda f: _assert(f.axes[0].get_yscale() == "log", "log scale")))
check("plot_loss(theme='dark')", lambda: one_axes(net.plot_loss(theme="dark")))
check("plot_architecture", lambda: one_axes(
    net.plot_architecture(),
    artists=lambda f: _assert(len(f.axes[0].collections) >= 3, "a node group per column")))
check("plot_architecture truncates wide layers",
      lambda: one_axes(wide_net().plot_architecture(max_nodes=4)))
check("plot_weight_distribution", lambda: one_axes(net.plot_weight_distribution(), expected=2))
check("plot_weight_heatmap(0)", lambda: one_axes(net.plot_weight_heatmap(0), expected=2))
check("plot_weight_heatmap(-1)", lambda: one_axes(net.plot_weight_heatmap(-1), expected=2))
check("plot_decision_boundary (binary)", lambda: one_axes(
    net.plot_decision_boundary(XOR_X, XOR_Y, resolution=40), expected=2))
check("plot_decision_boundary without labels", lambda: one_axes(
    net.plot_decision_boundary(XOR_X, resolution=40), expected=2))
check("plot_predictions (auto -> confusion)", lambda: one_axes(
    net.plot_predictions(XOR_X, XOR_Y),
    artists=lambda f: _assert(len(f.axes[0].images) == 1, "a matrix image")))
check("plot_loss_landscape", lambda: one_axes(
    net.plot_loss_landscape(XOR_X, XOR_Y, resolution=8, seed=0),
    artists=lambda f: _assert(hasattr(f.axes[0], "plot_surface"), "a 3-D axes")))
check("plot_summary", lambda: one_axes(net.plot_summary(XOR_X, XOR_Y), expected=4))
check("plot_summary without data", lambda: one_axes(net.plot_summary(), expected=3))

print("\nmulticlass and regression paths:")


def multiclass():
    gradiex.seed(3)
    m = gradiex.Network(0.1, batch_size=2)
    m.add_hidden(6, input_size=2, activation="tanh", init="xavier")
    m.add_output(3, activation="identity", loss="softmax_cross_entropy", init="xavier")
    m.build()
    X = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    Y = np.eye(3)[[0, 1, 2, 1]]
    m.train(X, Y, epochs=300, shuffle=True, seed=1)
    one_axes(m.plot_decision_boundary(X, Y, resolution=40))
    one_axes(m.plot_predictions(X, Y))


def regression():
    gradiex.seed(11)
    m = gradiex.Network(0.05, batch_size=8)
    m.add_hidden(8, input_size=1, activation="tanh", init="xavier")
    m.add_output(1, activation="identity", loss="mse", init="xavier")
    m.build()
    X = np.linspace(-1, 1, 30).reshape(-1, 1)
    Y = (X ** 2).reshape(-1, 1)
    m.train(X, Y, epochs=400, shuffle=True, seed=2)
    one_axes(m.plot_predictions(X, Y, kind="regression"),
             artists=lambda f: _assert(len(f.axes[0].collections) == 1, "a scatter"))


check("multiclass boundary + confusion matrix", multiclass)
check("regression predicted-vs-actual", regression)

print("\nloss landscape:")


def landscape_restores_weights():
    before = [np.array(net.weights(i)) for i in range(2)]
    plt.close(net.plot_loss_landscape(XOR_X, XOR_Y, resolution=6, seed=1))
    after = [np.array(net.weights(i)) for i in range(2)]
    for b, a in zip(before, after):
        _assert(np.array_equal(b, a), "the sweep left the network perturbed")


def landscape_restores_after_failure():
    # The restore sits in a finally; a mid-sweep error must not strand the
    # weights somewhere on the grid.
    before = [np.array(net.weights(i)) for i in range(2)]
    try:
        net.plot_loss_landscape(XOR_X, XOR_Y[:2], resolution=6)
    except ValueError:
        pass
    after = [np.array(net.weights(i)) for i in range(2)]
    for b, a in zip(before, after):
        _assert(np.array_equal(b, a), "weights not restored after a failed sweep")


def landscape_is_seeded():
    # The z limits are set from the surface's own min and max, so they stand in
    # for the whole slice without reaching into matplotlib's private artists.
    def extent(seed):
        fig = net.plot_loss_landscape(XOR_X, XOR_Y, resolution=6, seed=seed)
        limits = fig.axes[0].get_zlim()
        plt.close(fig)
        return limits

    _assert(extent(3) == extent(3), "same seed gave a different slice")
    _assert(extent(3) != extent(4), "different seeds gave the same slice")


def landscape_directions_are_filter_normalised():
    # Each direction row must carry the norm of the weight row it perturbs;
    # that normalisation is the whole reason the picture means anything.
    from gradiex.plotting import _filter_normalised_direction
    weights = np.array([[3.0, 4.0], [0.6, 0.8], [0.0, 0.0]], dtype=np.float32)
    d = _filter_normalised_direction(weights, np.random.default_rng(0))
    _assert(abs(np.linalg.norm(d[0]) - 5.0) < 1e-4, "row 0 norm not matched")
    _assert(abs(np.linalg.norm(d[1]) - 1.0) < 1e-4, "row 1 norm not matched")
    # An all-zero weight row has no scale to match, so it must not move.
    _assert(np.allclose(d[2], 0.0), "a zero weight row was given a direction")


def landscape_centre_is_the_trained_loss():
    _assert(abs(net.evaluate(XOR_X, XOR_Y)
                - float(np.mean([net.loss(x, y) for x, y in zip(XOR_X, XOR_Y)]))) < 1e-6,
            "evaluate disagrees with a per-row loop")


check("weights are restored after the sweep", landscape_restores_weights)
check("weights are restored after a failed sweep", landscape_restores_after_failure)
check("the slice is reproducible from its seed", landscape_is_seeded)
check("directions are filter-normalised", landscape_directions_are_filter_normalised)
check("evaluate matches a per-row loss loop", landscape_centre_is_the_trained_loss)
check("plot_loss_landscape(log=True)", lambda: one_axes(
    net.plot_loss_landscape(XOR_X, XOR_Y, resolution=6, seed=0, log=True)))
check("plot_loss_landscape(contours=False)", lambda: one_axes(
    net.plot_loss_landscape(XOR_X, XOR_Y, resolution=6, seed=0, contours=False)))
check("plot_loss_landscape(theme='dark')", lambda: one_axes(
    net.plot_loss_landscape(XOR_X, XOR_Y, resolution=6, seed=0, theme="dark")))

print("\ncomposition into caller-owned axes:")


def into_existing_axes():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    net.plot_loss(ax=axes[0])
    net.plot_architecture(ax=axes[1])
    _assert(len(fig.axes) == 2, "the helpers must not add axes when given one")
    plt.close(fig)


check("ax= is respected", into_existing_axes)
check("nothing is shown without show=True",
      lambda: _assert(matplotlib.get_backend().lower() == "agg"))

print("\nguards:")
expect_raises("empty history", lambda: gradiex.plot_loss([]), ValueError)
expect_raises("log with a non-positive loss",
              lambda: gradiex.plot_loss([1.0, 0.0], log=True), ValueError)
expect_raises("mismatched val_history",
              lambda: gradiex.plot_loss([1.0, 0.5], val_history=[1.0]), ValueError)
expect_raises("unknown theme", lambda: net.plot_loss(theme="neon"), ValueError)
expect_raises("plot_loss with no history at all",
              lambda: untrained_net().plot_loss(), ValueError)
expect_raises("weights before build",
              lambda: gradiex.plot_weight_distribution(unbuilt_net()), RuntimeError)
expect_raises("boundary needs 2 features",
              lambda: wide_net().plot_decision_boundary(np.zeros((3, 5))), ValueError)
expect_raises("boundary X must be (n, 2)",
              lambda: net.plot_decision_boundary(np.zeros((3, 3))), ValueError)
expect_raises("predictions row mismatch",
              lambda: net.plot_predictions(XOR_X, XOR_Y[:2]), ValueError)
expect_raises("predictions needs 2-D Y",
              lambda: net.plot_predictions(XOR_X, np.zeros(4)), ValueError)
expect_raises("unknown kind",
              lambda: net.plot_predictions(XOR_X, XOR_Y, kind="nope"), ValueError)
expect_raises("landscape resolution < 2",
              lambda: net.plot_loss_landscape(XOR_X, XOR_Y, resolution=1), ValueError)
expect_raises("landscape span <= 0",
              lambda: net.plot_loss_landscape(XOR_X, XOR_Y, span=0.0), ValueError)
expect_raises("landscape before build",
              lambda: gradiex.plot_loss_landscape(unbuilt_net(), XOR_X, XOR_Y), RuntimeError)
expect_raises("landscape needs a 3-D axes",
              lambda: net.plot_loss_landscape(
                  XOR_X, XOR_Y, resolution=4, ax=plt.subplots()[1]), ValueError)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for label, exc in failures:
        print(f"  {label}: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
print("ALL PLOTTING CHECKS PASSED")
