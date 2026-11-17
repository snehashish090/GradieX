# One script, run by cibuildwheel's test-command on every platform after the
# wheel is installed. test-command executes from a temporary directory outside
# the source tree, so `import gradiex` resolves to the INSTALLED wheel, not the
# checkout. (See [tool.cibuildwheel] in pyproject.toml.)
#
# This file exists because the inline `python -c "..."` this replaced could not
# survive Windows: backslash-escaped quotes are a /bin/sh idiom that cmd.exe
# passes through literally, so the whole command became a SyntaxError. A script
# referenced through the {project} placeholder needs no nested quoting on any
# shell.
#
# Keep it self-contained and dependency-light: it runs on cp39-cp314, on Linux,
# macOS and Windows. A wheel that imports but does not train -- or whose
# plotting layer raises -- must fail this check.
import sys

import numpy as np
import gradiex


def main() -> int:
    gradiex.seed(42)
    net = gradiex.Network(learning_rate=0.5, batch_size=4)
    net.add_hidden(8, input_size=2, activation="tanh", init="xavier")
    net.add_output(1, activation="sigmoid", loss="mse")
    net.build()

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    history = net.train(X, Y, epochs=2000, shuffle=True, seed=42)
    assert history[-1] < 0.01, history[-1]

    # Run the full Agg render path without touching the filesystem, so a wheel
    # whose plots raise is not a usable wheel. matplotlib is a hard dependency
    # and is installed via test-requires.
    gradiex.plot_loss(history, show=False).canvas.draw()
    print("XOR converged:", history[0], "->", history[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
