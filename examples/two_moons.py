#!/usr/bin/env python3
"""A worked example: binary classification on two interleaving moons.

Run it from the release root, after building the extension in place:

    python3 setup.py build_ext --inplace
    python3 examples/two_moons.py

It trains a 2 -> 16 -> 16 -> 1 network and writes every diagram
gradiex.plotting offers into examples/plots/.

Two moons rather than XOR because the boundary is genuinely curved: a linear
model cannot separate the classes, so the decision-boundary plot shows the
hidden layers doing real work instead of a four-point toy.
"""

import pathlib
import sys

# Import gradiex from this release rather than whatever is installed.
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")          # write files, never open a window
import numpy as np

import gradiex

PLOTS = pathlib.Path(__file__).resolve().parent / "plots"
DATA_SEED = 7
INIT_SEED = 42
SHUFFLE_SEED = 1


# --------------------------------------------------------------------- data

def make_moons(n_samples, noise, rng):
    """Two interleaving half-circles. No sklearn; the engine has no dependencies
    and neither should its example."""
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    t_outer = np.linspace(0.0, np.pi, n_outer)
    t_inner = np.linspace(0.0, np.pi, n_inner)
    outer = np.stack([np.cos(t_outer), np.sin(t_outer)], axis=1)
    inner = np.stack([1.0 - np.cos(t_inner), 0.5 - np.sin(t_inner)], axis=1)

    X = np.vstack([outer, inner])
    y = np.hstack([np.zeros(n_outer), np.ones(n_inner)])
    X += rng.normal(0.0, noise, X.shape)
    return X, y


def split(X, y, train_fraction, rng):
    order = rng.permutation(len(X))
    cut = int(len(X) * train_fraction)
    train, test = order[:cut], order[cut:]
    return X[train], y[train], X[test], y[test]


def accuracy(net, X, y):
    preds = np.array([net.predict(row)[0] for row in X])
    return float(((preds >= 0.5).astype(float) == y.ravel()).mean())


def build():
    """2 -> 16 tanh -> 16 tanh -> 1 sigmoid, trained against binary cross-entropy."""
    net = gradiex.Network(learning_rate=0.1, batch_size=16)
    net.add_hidden(16, input_size=2, activation="tanh", init="xavier")
    net.add_hidden(16, activation="tanh", init="xavier")
    net.add_output(1, activation="sigmoid", loss="bce", init="xavier")
    net.build()
    return net


def save(fig, name, theme="light"):
    path = PLOTS / name
    fig.savefig(path, dpi=130, facecolor=gradiex.THEMES[theme]["surface"])
    matplotlib.pyplot.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(DATA_SEED)

    X, y = make_moons(400, noise=0.13, rng=rng)
    # Standardise: the features are already on a similar scale, but centring
    # keeps tanh off its flat tails at initialisation.
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X_train, y_train, X_test, y_test = split(X, y, 0.75, rng)
    Y_train = y_train.reshape(-1, 1)
    Y_test = y_test.reshape(-1, 1)

    print(f"gradiex {gradiex.__version__} ({gradiex.__dtype__} engine)")
    print(f"dataset: {len(X_train)} train / {len(X_test)} test, 2 features, 2 classes")

    # ---- the main run: one engine call, loss curve drawn on the way out ----
    gradiex.seed(INIT_SEED)
    net = build()
    print(f"\nmodel:  {net}")

    EPOCHS = 600
    history = net.train(X_train, Y_train, epochs=EPOCHS, shuffle=True,
                        seed=SHUFFLE_SEED, plot=True, smooth=25)
    print(f"\ntrained {EPOCHS} epochs: loss {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"  train accuracy {accuracy(net, X_train, Y_train):.1%}")
    print(f"  test  accuracy {accuracy(net, X_test, Y_test):.1%}")

    print(f"\nplots -> {PLOTS.relative_to(ROOT)}/")
    save(net.last_figure, "01_loss.png")
    save(net.plot_loss(log=True, title="Training loss (log scale)"), "02_loss_log.png")
    save(net.plot_loss(theme="dark"), "03_loss_dark.png", theme="dark")
    save(net.plot_architecture(), "04_architecture.png")
    save(net.plot_weight_distribution(), "05_weight_distribution.png")
    save(net.plot_weight_heatmap(0, title="hidden 0 weights"), "06_weights_hidden0.png")
    save(net.plot_decision_boundary(X_test, Y_test,
                                    title="Decision boundary (test set)"),
         "07_decision_boundary.png")
    save(net.plot_predictions(X_test, Y_test,
                              title="Confusion matrix (test set)"),
         "08_confusion_matrix.png")
    save(net.plot_loss_landscape(X_train, Y_train, seed=0, resolution=40),
         "09_loss_landscape.png")
    save(net.plot_loss_landscape(X_train, Y_train, seed=0, resolution=40, log=True,
                                 title="Loss landscape (log scale)"),
         "10_loss_landscape_log.png")
    save(net.plot_summary(X_test, Y_test), "11_summary.png")
    save(net.plot_summary(X_test, Y_test, theme="dark"), "12_summary_dark.png",
         theme="dark")

    # ---- a monitored run, to show train against held-out loss ----
    # train() runs every epoch inside the engine, so watching a validation split
    # means calling it one epoch at a time. Each call is still an engine call;
    # nothing about the update moves into Python.
    gradiex.seed(INIT_SEED)
    watched = build()
    train_curve, test_curve = [], []
    for epoch in range(EPOCHS):
        train_curve.append(
            watched.train(X_train, Y_train, epochs=1, shuffle=True,
                          seed=SHUFFLE_SEED + epoch)[0])
        test_curve.append(
            float(np.mean([watched.loss(x, t) for x, t in zip(X_test, Y_test)])))

    save(gradiex.plot_loss(train_curve, val_history=test_curve,
                           labels=("train", "held-out"),
                           title="Train vs held-out loss"),
         "13_train_vs_holdout.png")
    print(f"\nmonitored run: train {train_curve[-1]:.4f}, held-out {test_curve[-1]:.4f}"
          f"  (gap {test_curve[-1] - train_curve[-1]:+.4f})")
    print(f"  test accuracy {accuracy(watched, X_test, Y_test):.1%}")

    print(f"\n{len(list(PLOTS.glob('*.png')))} plots saved.")


if __name__ == "__main__":
    main()
