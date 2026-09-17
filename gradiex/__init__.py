"""GradieX -- a dependency-free feedforward neural network engine in C++.

The engine lives in the compiled extension :mod:`gradiex._core`; this package
re-exports it and adds a matplotlib-backed plotting layer. Training itself never
touches Python: :meth:`Network.train` hands the whole dataset to
``core::Network::train`` and gets a loss history back.

    import gradiex

    gradiex.seed(42)                       # reproducible weight init
    net = gradiex.Network(learning_rate=0.5, batch_size=4)
    net.add_hidden(8, input_size=2, activation="tanh", init="xavier")
    net.add_output(1, activation="sigmoid", loss="mse")
    net.build()

    history = net.train(X, Y, epochs=4000, shuffle=True, seed=42, plot=True)
    net.plot_summary(X, Y, show=True)
"""

from __future__ import annotations

from . import _core
from ._core import activations, initializers, losses, seed

from .plotting import (
    THEMES,
    plot_architecture,
    plot_decision_boundary,
    plot_loss,
    plot_loss_landscape,
    plot_predictions,
    plot_weight_distribution,
    plot_weight_heatmap,
    summary,
)

__version__ = _core.__version__
__author__ = _core.__author__
__dtype__ = _core.__dtype__

__all__ = [
    "Network",
    "activations",
    "losses",
    "initializers",
    "seed",
    "THEMES",
    "plot_loss",
    "plot_architecture",
    "plot_weight_distribution",
    "plot_weight_heatmap",
    "plot_decision_boundary",
    "plot_predictions",
    "plot_loss_landscape",
    "summary",
    "__version__",
    "__author__",
    "__dtype__",
]


class Network(_core.Network):
    """A feedforward network, with plotting attached.

    Identical to :class:`gradiex._core.Network` except that :meth:`train`
    remembers the loss history it returned and can draw it, and the ``plot_*``
    methods forward to :mod:`gradiex.plotting`.

    Layers are declared, then built:

        net = gradiex.Network(learning_rate=0.5, batch_size=4)
        net.add_hidden(8, input_size=2, activation="tanh", init="xavier")
        net.add_output(1, activation="sigmoid", loss="mse")
        net.build()

    ``batch_size`` is how many samples are accumulated before one update:
    1 is per-sample SGD, 0 means one batch per epoch.
    """

    #: Loss history from the most recent :meth:`train` call, or ``None``.
    history = None

    def train(self, X, Y, epochs=100, shuffle=False, seed=None, verbose=False,
              log_every=0, plot=False, **plot_kwargs):
        """Train in the engine and return the mean loss of each epoch.

        Beyond the engine's own arguments:

        plot : bool
            Draw the loss curve when training finishes. The figure is returned
            on :attr:`last_figure` and shown only if ``show=True`` is passed
            through.
        **plot_kwargs
            Forwarded to :func:`gradiex.plotting.plot_loss` (``smooth``, ``log``,
            ``theme``, ``title``, ``show``, ...).
        """
        history = super().train(X, Y, epochs, shuffle, seed, verbose, log_every)
        self.history = history
        self.last_figure = None
        if plot:
            if not history:
                raise ValueError(
                    "plot=True but training produced no history; epochs was 0 or the "
                    "dataset was empty")
            self.last_figure = plot_loss(history, **plot_kwargs)
        elif plot_kwargs:
            raise TypeError(
                f"train() got plotting arguments {sorted(plot_kwargs)} without plot=True")
        return history

    # -- plotting, all thin forwards so the functions stay usable standalone

    def plot_loss(self, history=None, **kwargs):
        """Plot a loss history, defaulting to the one from the last ``train()``."""
        if history is None:
            history = self.history
        if not history:
            raise ValueError("no loss history: pass one, or call train() first")
        return plot_loss(history, **kwargs)

    def plot_architecture(self, **kwargs):
        return plot_architecture(self, **kwargs)

    def plot_weight_distribution(self, **kwargs):
        return plot_weight_distribution(self, **kwargs)

    def plot_weight_heatmap(self, layer=-1, **kwargs):
        return plot_weight_heatmap(self, layer, **kwargs)

    def plot_decision_boundary(self, X, y=None, **kwargs):
        return plot_decision_boundary(self, X, y, **kwargs)

    def plot_predictions(self, X, Y, **kwargs):
        return plot_predictions(self, X, Y, **kwargs)

    def plot_loss_landscape(self, X, Y, **kwargs):
        """The loss surface on a random 2-D slice through the trained weights."""
        return plot_loss_landscape(self, X, Y, **kwargs)

    def plot_summary(self, X=None, Y=None, history=None, **kwargs):
        """One figure covering architecture, loss, weights and the data fit."""
        if history is None:
            history = self.history
        return summary(self, history, X, Y, **kwargs)
