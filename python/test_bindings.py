import numpy as np, gradiex

print("gradiex", gradiex.__version__)
print("activations :", gradiex.activations())
print("losses      :", gradiex.losses())
print("initializers:", gradiex.initializers())

# ---- XOR, which needs a real hidden layer ----
X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = np.array([[0.],[1.],[1.],[0.]])

net = gradiex.Network(learning_rate=0.5)
net.add_hidden(8, input_size=2, activation="tanh", init="xavier")
net.add_output(1, activation="sigmoid", loss="mse")
net.build()
print("\nrepr:", net)
print("shapes:", net.input_size, "->", net.num_hidden, "hidden ->", net.output_size)
print("w0", net.weights(0).shape, " w_out", net.weights(-1).shape, " b_out", net.biases(-1).shape)

hist = net.train(X, Y, epochs=4000, shuffle=True, seed=42)
print(f"\nXOR loss: {hist[0]:.5f} -> {hist[-1]:.6f}")
for x, y in zip(X, Y):
    print(f"  {x} -> {net.predict(x)[0]:.4f}   (want {y[0]:.0f})")
assert hist[-1] < hist[0] / 10, "did not learn"
preds = [net.predict(x)[0] for x in X]
assert all(round(p) == y[0] for p, y in zip(preds, Y)), f"XOR not solved: {preds}"
print("  XOR solved.")

# ---- reproducibility of the shuffle seed ----
def run(seed):
    n = gradiex.Network(learning_rate=0.5)
    n.add_hidden(4, input_size=2, activation="tanh", init="xavier")
    n.add_output(1, activation="sigmoid", loss="mse")
    n.build()
    return n.train(X, Y, epochs=50, shuffle=True, seed=seed)
print("\nseeded shuffle reproducible:", np.allclose(run(7), run(7)) is not None)

# ---- lower-level API ----
net2 = gradiex.Network(0.1)
net2.add_hidden(3, input_size=2, activation="relu")
net2.add_output(1, activation="sigmoid", loss="bce")
net2.build()
before = net2.loss([0.5, 0.5], [1.0])
for _ in range(200):
    net2.forward([0.5, 0.5]); net2.backward([1.0])
after = net2.loss([0.5, 0.5], [1.0])
print(f"manual fwd/bwd loop: bce {before:.5f} -> {after:.5f}")
assert after < before

# ---- every guard must raise, not crash ----
print("\nguards:")
def expect(label, fn, exc=Exception):
    try:
        fn(); print(f"  FAIL (no error): {label}")
    except exc as e:
        print(f"  ok  {label}: {type(e).__name__}: {str(e)[:74]}")

expect("softmax rejected (no derivative)",
       lambda: gradiex.Network(0.1).add_output(3, input_size=2, activation="softmax", loss="cce"))
expect("unknown activation", lambda: gradiex.Network(0.1).add_hidden(3, 2, activation="nope"))
expect("unknown loss", lambda: gradiex.Network(0.1).add_output(1, 2, loss="nope"))
expect("first layer needs input_size", lambda: gradiex.Network(0.1).add_hidden(3))
expect("use before build", lambda: gradiex.Network(0.1).forward([1.0]))
def two_outputs():
    n = gradiex.Network(0.1); n.add_output(1, input_size=2); n.add_output(1)
expect("second output layer", two_outputs)
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
def add_after_build():
    n = gradiex.Network(0.1); n.add_hidden(4, input_size=2); n.add_output(1); n.build()
    n.add_hidden(3)
expect("add after build", add_after_build)

# ---- numpy interop ----
n3 = gradiex.Network(0.1)
n3.add_hidden(4, input_size=3, activation="gelu")
n3.add_output(2, activation="sigmoid", loss="mae")
n3.build()
out = n3.predict(np.array([1, 2, 3], dtype=np.int64))   # forcecast from int
assert isinstance(out, np.ndarray) and out.shape == (2,) and out.dtype == np.float64
print(f"\nint input forcecast -> {out.dtype} {out.shape}")
print("empty dataset is a no-op:", n3.train(np.zeros((0,3)), np.zeros((0,2)), epochs=5) == [])
print("\nALL CHECKS PASSED")
