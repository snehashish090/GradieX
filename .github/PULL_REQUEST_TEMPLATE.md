## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Checklist

- [ ] `./test_gradiex` prints `112 passed, 0 failed` and exits 0
- [ ] Zero warnings under `-Wall -Wextra -pedantic`
- [ ] Clean under `-fsanitize=address,undefined`
- [ ] New behaviour has a test
- [ ] Docs updated (`main.cpp` header and `README.md`) if the API changed
- [ ] `CHANGELOG.md` entry added

## If this touches the backward pass

- [ ] The gradient check in Test 1 still passes
- [ ] A configuration was added to Test 1's `cases` list for any new activation,
      loss, or path through `accumulate_gradients`
- [ ] None of the invariants in `CONTRIBUTING.md` were broken

<!-- If you changed the model file format, confirm the version was bumped. -->
