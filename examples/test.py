import gradiex
import time 
import random 
import numpy as np
def save(fig, name, theme="light"):
    path = f"./{name}"
    fig.savefig(path, dpi=130, facecolor=gradiex.THEMES[theme]["surface"])
    return path

x = [np.random.rand(10000) for i in range(10)]
y = [np.random.rand(20) for i in range(10)]
t1 = time.time()
# 1. Seed RNG
gradiex.seed(42)
print("building...")
# 3. Build Network
net = gradiex.Network(learning_rate=0.01, batch_size=32)  
for i in range(100):
    net.add_hidden(10000, input_size=10000, activation="leaky_relu", init="xavier")
net.add_output(20, activation="sigmoid", loss="bce")
net.build()
t2 = time.time()
print("Network build time:", t2 - t1)
net.train(x, y, epochs=10)
t3 = time.time()
print("Training time:", t3 - t2)
op = net.plot_architecture()
save(op, "architecture.png")
t4 = time.time()
print("Plotting time:", t4 - t3)

t5 = time.time()
loss_curve = net.plot_loss()
save(loss_curve, "loss_curve.png")
t6 = time.time()
print("Loss curve plotting time:", t6 - t5)

# 5. Evaluate and Plot Loss Curve