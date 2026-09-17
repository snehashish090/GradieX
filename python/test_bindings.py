# The extension is built in-place at the release root, one level up from this file.
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np, gradiex

print("gradiex", gradiex.__version__, "dtype", gradiex.__dtype__)
print("activations :", gradiex.activations())
print("losses      :", gradiex.losses())
print("initializers:", gradiex.initializers())

# ---- XOR, which needs a real hidden layer ----
X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = np.array([[0.],[1.],[1.],[0.]])

gradiex.seed(42)
net = gradiex.Network(learning_rate=0.5, batch_size=4)
net.add_hidden(8, input_size=2, activation="tanh", init="xavier")
net.add_output(1, activation="sigmoid", loss="mse")
net.build()
print("\nrepr:", net)
print("shapes:", net.input_size, "->", net.num_hidden, "hidden ->", net.output_size)
print("w0", net.weights(0).shape, net.weights(0).dtype,
      " w_out", net.weights(-1).shape, " b_out", net.biases(-1).shape)

hist = net.train(X, Y, epochs=4000, shuffle=True, seed=42)
print(f"\nXOR loss: {hist[0]:.5f} -> {hist[-1]:.6f}")
for x, y in zip(X, Y):
    print(f"  {x} -> {net.predict(x)[0]:.4f}   (want {y[0]:.0f})")
assert len(hist) == 4000, f"history should hold one entry per epoch, got {len(hist)}"
assert hist[-1] < hist[0] / 10, "did not learn"
preds = [net.predict(x)[0] for x in X]
assert all(round(p) == y[0] for p, y in zip(preds, Y)), f"XOR not solved: {preds}"
print("  XOR solved.")

# ---- reproducibility: same init seed + same shuffle seed => identical history ----
def run(init_seed, shuffle_seed):
    gradiex.seed(init_seed)
    n = gradiex.Network(learning_rate=0.5, batch_size=2)
    n.add_hidden(4, input_size=2, activation="tanh", init="xavier")
    n.add_output(1, activation="sigmoid", loss="mse")
    n.build()
    return n.train(X, Y, epochs=50, shuffle=True, seed=shuffle_seed)

assert run(7, 7) == run(7, 7), "seeded run is not reproducible"
assert run(7, 7) != run(8, 7), "a different init seed should change the run"
print("\nseeded init + shuffle reproducible:", True)

# ---- batch_size drives the engine's gradient accumulation ----
gradiex.seed(1)
full = gradiex.Network(0.5, batch_size=0)      # 0 == one batch per epoch
full.add_hidden(4, input_size=2, activation="tanh", init="xavier")
full.add_output(1, activation="sigmoid", loss="mse")
full.build()
h_full = full.train(X, Y, epochs=200)
gradiex.seed(1)
per = gradiex.Network(0.5, batch_size=1)       # per-sample SGD
per.add_hidden(4, input_size=2, activation="tanh", init="xavier")
per.add_output(1, activation="sigmoid", loss="mse")
per.build()
h_per = per.train(X, Y, epochs=200)
assert h_full[-1] < h_full[0] and h_per[-1] < h_per[0]
assert h_full != h_per, "batch_size should change the update schedule"
print(f"batch=0 (full) {h_full[0]:.5f} -> {h_full[-1]:.5f}   "
      f"batch=1 (sgd) {h_per[0]:.5f} -> {h_per[-1]:.5f}")

# ---- lower-level API: the engine primitives, one call each ----
net2 = gradiex.Network(0.1)
net2.add_hidden(3, input_size=2, activation="relu")
net2.add_output(1, activation="sigmoid", loss="bce")
net2.build()
before = net2.loss([0.5, 0.5], [1.0])
for _ in range(200):
    net2.forward([0.5, 0.5]); net2.backward([1.0])
after = net2.loss([0.5, 0.5], [1.0])
print(f"\nmanual fwd/bwd loop: bce {before:.5f} -> {after:.5f}")
assert after < before

# zero / accumulate / scale / apply, hand-rolled against train_batch
net3 = gradiex.Network(0.2)
net3.add_hidden(3, input_size=2, activation="tanh", init="xavier")
net3.add_output(1, activation="sigmoid", loss="mse")
net3.build()
b0 = net3.loss([0.3, 0.7], [1.0])
for _ in range(100):
    net3.zero_gradients()
    for x, y in ([0.3, 0.7], [1.0]), ([0.7, 0.3], [0.0]):
        net3.forward(x); net3.accumulate(y)
    net3.scale_gradients(0.5)
    net3.apply_gradients()
b1 = net3.loss([0.3, 0.7], [1.0])
print(f"manual accumulate loop: mse {b0:.5f} -> {b1:.5f}")
assert b1 < b0

# train_batch is the same thing in one engine call
net4 = gradiex.Network(0.2)
net4.add_hidden(3, input_size=2, activation="tanh", init="xavier")
net4.add_output(1, activation="sigmoid", loss="mse")
net4.build()
first = net4.train_batch(X, Y)
for _ in range(400):
    last = net4.train_batch(X, Y)
print(f"train_batch loop:       mse {first:.5f} -> {last:.5f}")
assert last < first

# ---- softmax is reachable through the fused identity + softmax_cross_entropy path ----
gradiex.seed(3)
clf = gradiex.Network(0.1, batch_size=2)
clf.add_hidden(6, input_size=2, activation="tanh", init="xavier")
clf.add_output(3, activation="identity", loss="softmax_cross_entropy", init="xavier")
clf.build()
Xc = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
Yc = np.eye(3)[[0, 1, 2, 1]]
hc = clf.train(Xc, Yc, epochs=1500, shuffle=True, seed=1)
logits = np.array([clf.predict(x) for x in Xc])
acc = (logits.argmax(1) == Yc.argmax(1)).mean()
print(f"\nfused softmax+CE: loss {hc[0]:.5f} -> {hc[-1]:.5f}, accuracy {acc:.2f}")
assert hc[-1] < hc[0] and acc == 1.0

# ---- evaluate(): dataset-wide loss, engine-side ----
ev = gradiex.Network(0.1)
ev.add_hidden(4, input_size=2, activation="tanh", init="xavier")
ev.add_output(1, activation="sigmoid", loss="mse", init="xavier")
ev.build()
manual = float(np.mean([ev.loss(x, y) for x, y in zip(X, Y)]))
assert abs(ev.evaluate(X, Y) - manual) < 1e-6, "evaluate disagrees with a per-row loop"
print(f"\nevaluate() == mean of loss(): {ev.evaluate(X, Y):.6f}")
before_eval = [np.array(ev.weights(i)) for i in range(2)]
ev.evaluate(X, Y)
assert all(np.array_equal(a, np.array(ev.weights(i))) for i, a in enumerate(before_eval)), \
    "evaluate() must not update anything"
print("evaluate() leaves the weights alone: True")

# ---- set_weights / set_biases round-trip ----
w0 = np.array(ev.weights(0))
ev.set_weights(0, np.zeros_like(w0))
assert np.all(np.array(ev.weights(0)) == 0.0)
ev.set_weights(0, w0)
assert np.array_equal(np.array(ev.weights(0)), w0), "weights did not round-trip"
ev.set_weights(0, w0.ravel())                     # flat is accepted too
assert np.array_equal(np.array(ev.weights(0)), w0)
b0 = np.array(ev.biases(0))
ev.set_biases(0, np.ones_like(b0))
assert np.all(np.array(ev.biases(0)) == 1.0)
ev.set_biases(0, b0)
assert np.array_equal(np.array(ev.biases(0)), b0), "biases did not round-trip"
print("set_weights / set_biases round-trip:", True)

# ---- every guard must raise, not crash ----
print("\nguards:")
def expect(label, fn, exc=Exception):
    try:
        fn(); print(f"  FAIL (no error): {label}")
    except exc as e:
        print(f"  ok  {label}: {type(e).__name__}: {str(e)[:74]}")

expect("softmax activation rejected (Jacobian, not implemented)",
       lambda: gradiex.Network(0.1).add_output(3, input_size=2, activation="softmax", loss="cce"))
expect("unknown activation", lambda: gradiex.Network(0.1).add_hidden(3, 2, activation="nope"))
expect("unknown loss", lambda: gradiex.Network(0.1).add_output(1, 2, loss="nope"))
expect("first layer needs input_size", lambda: gradiex.Network(0.1).add_hidden(3))
expect("use before build", lambda: gradiex.Network(0.1).forward([1.0]))
expect("negative batch_size", lambda: gradiex.Network(0.1, batch_size=-1))
expect("non-positive learning_rate", lambda: gradiex.Network(0.0))
def two_outputs():
    n = gradiex.Network(0.1); n.add_output(1, input_size=2); n.add_output(1)
expect("second output layer", two_outputs)
def hidden_after_output():
    n = gradiex.Network(0.1); n.add_output(1, input_size=2); n.add_hidden(3)
expect("hidden after output", hidden_after_output)
def bad_chain():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.add_hidden(3, input_size=99)
expect("input_size mismatch", bad_chain)
def build_no_output():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.build()
expect("build without output", build_no_output)
def wrong_x():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2)
    n.add_output(1); n.build(); n.forward([1.0, 2.0, 3.0])
expect("wrong input length", wrong_x)
def bad_layer_idx():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.add_output(1); n.build()
    n.weights(17)
expect("layer index out of range", bad_layer_idx)
def mismatched_rows():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.add_output(1); n.build()
    n.train(np.zeros((4,2)), np.zeros((3,1)), epochs=1)
expect("X/Y row mismatch", mismatched_rows)
def ragged_row():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.add_output(1); n.build()
    n.train(np.zeros((4,3)), np.zeros((4,1)), epochs=1)
expect("X row width mismatch", ragged_row)
def one_d_X():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.add_output(1); n.build()
    n.train(np.zeros(4), np.zeros((4,1)), epochs=1)
expect("X must be 2-D", one_d_X)
def add_after_build():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.add_output(1); n.build()
    n.add_hidden(3)
expect("add after build", add_after_build)
def eval_row_mismatch():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.add_output(1); n.build()
    n.evaluate(np.zeros((4, 2)), np.zeros((3, 1)))
expect("evaluate row mismatch", eval_row_mismatch)
def eval_empty():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.add_output(1); n.build()
    n.evaluate(np.zeros((0, 2)), np.zeros((0, 1)))
expect("evaluate on an empty dataset", eval_empty)
def set_weights_wrong_size():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.add_output(1); n.build()
    n.set_weights(0, np.zeros(7))
expect("set_weights wrong size", set_weights_wrong_size)
def set_biases_wrong_size():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.add_output(1); n.build()
    n.set_biases(0, np.zeros(9))
expect("set_biases wrong size", set_biases_wrong_size)
def set_weights_before_build():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.add_output(1)
    n.set_weights(0, np.zeros(8))
expect("set_weights before build", set_weights_before_build)
def build_twice():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.add_output(1); n.build(); n.build()
expect("build twice", build_twice)

# ---- numpy interop ----
n5 = gradiex.Network(0.1)
n5.add_hidden(4, input_size=3, activation="gelu")
n5.add_output(2, activation="sigmoid", loss="mae")
n5.build()
out = n5.predict(np.array([1, 2, 3], dtype=np.int64))   # forcecast from int
assert isinstance(out, np.ndarray) and out.shape == (2,) and out.dtype == np.float32
print(f"\nint input forcecast -> {out.dtype} {out.shape}")
print("gelu trains:", n5.train(np.random.rand(8,3), np.random.rand(8,2), epochs=20)[-1] > 0)
print("empty dataset is a no-op:", n5.train(np.zeros((0,3)), np.zeros((0,2)), epochs=5) == [])
print("zero epochs is a no-op:", n5.train(np.zeros((4,3)), np.zeros((4,2)), epochs=0) == [])

# ---- mutable properties reach the engine ----
n5.learning_rate = 0.05
n5.batch_size = 2
# The engine stores the rate as float32, so it reads back rounded, not bit-identical.
assert abs(n5.learning_rate - 0.05) < 1e-7 and n5.batch_size == 2
print("learning_rate / batch_size settable after build:", n5.learning_rate, n5.batch_size)

print("\nALL CHECKS PASSED")
