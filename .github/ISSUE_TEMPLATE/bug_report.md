---
name: Bug report
about: Something produces a wrong result, crashes, or will not build
title: ''
labels: bug
---

**What happened**

<!-- What you observed, and what you expected instead. -->

**Minimal reproduction**

<!-- The smallest main() that shows the problem. Include the seed: every
     GradieX run is reproducible from one, so this makes it exact. -->

```cpp
NeuralNetwork net(2, 2, 4, 1, "tanh", "sigmoid", "binary_cross_entropy", 42);
// ...
```

**Environment**

- Compiler and version: <!-- g++ --version -->
- OS:
- Build flags: <!-- e.g. -std=c++17 -O2 -->

**Test suite**

<!-- Does `./test_gradiex` still pass on your machine? Paste the summary line. -->

```
SUMMARY:  112 passed, 0 failed
```
