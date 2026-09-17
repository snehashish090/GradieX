# Examples

```sh
cd 0.0.3
python3 setup.py build_ext --inplace
python3 examples/two_moons.py
```

## `two_moons.py`

Binary classification on two interleaving half-circles — a curved boundary, so a
linear model cannot separate the classes and the hidden layers have to do real
work. 400 samples, split 300 train / 100 test, standardised.

```
2 → 16 tanh → 16 tanh → 1 sigmoid     binary cross-entropy
lr 0.1, batch 16, 600 epochs, shuffled
```

Result: loss `0.4291 → 0.0009`, **100% train and 100% test accuracy**. Everything
is seeded (`gradiex.seed` for weight init, `seed=` for the shuffle, a fixed
NumPy generator for the data), so the run reproduces exactly.

The script also does a second, epoch-stepped run to plot training against
held-out loss. `train()` runs every epoch inside the engine, so watching a
validation split means calling it one epoch at a time — each call is still an
engine call, and nothing about the update moves into Python.

## Plots

Written to `examples/plots/`.

| file | what it shows |
| --- | --- |
| `01_loss.png` | the loss curve `train(plot=True)` draws on its own, with a 25-epoch mean over the raw trace |
| `02_loss_log.png` | the same run on a log axis, where the late descent is actually visible |
| `03_loss_dark.png` | dark theme |
| `04_architecture.png` | the layer graph; the 16-wide layers are drawn truncated with `⋮` and labelled with their true width |
| `05_weight_distribution.png` | one histogram per layer with its σ — a layer collapsed at zero or spread far wider than its neighbours is the thing to look for |
| `06_weights_hidden0.png` | the first hidden layer's weight matrix, diverging about zero so the sign reads |
| `07_decision_boundary.png` | the learned boundary over the test set, with the 0.5 contour |
| `08_confusion_matrix.png` | test-set confusion matrix |
| `09_loss_landscape.png` | the loss surface on a random 2-D slice through the trained weights |
| `10_loss_landscape_log.png` | the same slice on a log axis — **the informative one here**: the linear version is a flat plain, the log version shows the funnel into the minimum |
| `11_summary.png` | architecture, loss, weights and boundary in one figure |
| `12_summary_dark.png` | the same, dark theme |
| `13_train_vs_holdout.png` | training against held-out loss from the monitored run |

### Reading the loss landscape

Two random directions are drawn through the trained weights and **filter-
normalised** — each direction row is scaled to the norm of the weight row it
perturbs. That normalisation is what makes the picture mean anything: a network's
loss is invariant to rescaling a neuron's weights, so an unnormalised step moves
small-weight neurons a long way in function space and large-weight ones barely at
all, and the result describes the weight scales rather than the loss. The method
is from Li et al., *Visualizing the Loss Landscape of Neural Nets* (2018).

It is a **random 2-D slice of a 337-dimensional surface**, so read it as a
texture, not a map. It shows whether the basin the optimiser settled in is broad
and smooth or narrow and ragged; it does not show the path taken or where other
minima are. A different `seed=` gives a different slice, and the sweep restores
the trained weights exactly when it finishes.

## A note on what this example caught

The first run of this script diverged to `NaN` around epoch 243, with the loss
already down at 0.003. The cause was in `core/loss.h`: the log-based losses
guarded their probabilities with `1e-12`, a double-precision constant that
rounds away entirely in a float32 engine — `1.0f - 1e-12f` *is* `1.0f`, so the
upper clamp never bit. Once the sigmoid saturated to exactly 1.0, the loss hit
`log(0) = -inf` and the derivative divided by zero; multiplied by an activation
derivative of 0, that is `inf * 0 = NaN`, which reaches every weight on the next
update and never washes out. The guard is now `1e-7f`, representable either side
of 1.0, and `tests/test_layer.cpp` section G pins it.
