# Releasing

How a GradieX release reaches PyPI. Everything is built by CI; nothing is uploaded
from a laptop.

The pipeline is [`.github/workflows/release.yml`](.github/workflows/release.yml). It
has three jobs:

| job | runs on | does |
| --- | --- | --- |
| `wheels` | tags, PRs that touch the engine or build, `workflow_dispatch` | one wheel per platform, each tested by training XOR before it is kept |
| `sdist` | same | builds the tarball, then installs it into a clean venv and imports it |
| `publish` | tags `v*` only | downloads every artifact and uploads to PyPI via trusted publishing |

Publishing therefore has exactly one trigger: pushing a tag that starts with `v`.

## One-time setup

These live in web UIs and cannot be done from the repository.

1. **GitHub environment.** Repository → Settings → Environments → create one named
   `pypi`. The `publish` job declares `environment: pypi`; if the environment does not
   exist the job can still run, but there is nothing to attach a protection rule or a
   secret to later. Adding required reviewers here is a good gate, since it pauses the
   upload until someone approves it.

2. **PyPI trusted publisher.** On PyPI → *Your projects* → *Publishing* (or
   <https://pypi.org/manage/account/publishing/>), add a trusted publisher with:

   | field | value |
   | --- | --- |
   | PyPI project name | `gradiex` |
   | Owner | `snehashish090` |
   | Repository name | `GradieX` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   The workflow name must match the filename exactly, and the environment name must
   match the GitHub environment. No API token is stored anywhere — the job mints a
   short-lived OIDC token, which is why it needs `id-token: write`.

   **Pending publishers.** If `gradiex` does not exist on PyPI yet, create a *pending*
   publisher instead. The first successful publish creates the project and converts the
   pending publisher into a normal one.

3. **Check the project name is still free.** <https://pypi.org/project/gradiex/>.
   PyPI names are first-come, case-insensitive and permanent: if someone has taken
   `gradiex`, the project must be renamed in `pyproject.toml` and
   `python/gradiex_module.cpp` before the first upload, not after.

## Cutting a release

Each release lives in its own directory (`0.0.3/`, `0.0.4/`, ...). Before tagging:

1. **The version must agree in three places.** They are not generated from one another:

   | file | what |
   | --- | --- |
   | `pyproject.toml` | `version = "0.0.3"` |
   | `python/gradiex_module.cpp` | `m.attr("__version__") = py::str("0.0.3")` |
   | `README.md` | the version badge |

   The sdist and the extension are built from these separately, so `gradiex.__version__`
   and the PyPI version can silently disagree. PyPI rejects a re-upload of an existing
   version, so a mismatch here is not a fixable typo — it burns the version number.

2. **CHANGELOG.md has a section for the version**, dated, following Keep a Changelog.

3. **Nothing stray ships in the sdist.** `setup.py` puts `core/*.h` into the sdist via
   `depends`, so *every* header in `core/` is published and must be complete. Check:

   ```bash
   python3 -m build --sdist
   tar tzf dist/gradiex-*.tar.gz | sort
   ```

   Files that are not part of the engine — scratch headers, benchmarks, datasets —
   should be removed from `core/` or excluded, not left to ship.

4. **Run the suites.** All four, from the release root:

   ```bash
   ./tests/run_tests.sh --simd     # C++ engine, 91 checks; --simd turns the
                                   # `#pragma omp simd` directives back on, which is
                                   # the only way a malformed one shows up
   ./tests/run_tests.sh --asan     # same, under ASan/UBSan
   python3 setup.py build_ext --inplace
   MPLBACKEND=Agg python3 python/test_bindings.py
   MPLBACKEND=Agg python3 python/test_plotting.py
   ```

   The Python checks are scripts, not pytest tests: they import `gradiex` from the
   release root and assert as they go. `tests/run_tests.sh` and the CI `test-command`
   are self-contained; `python/test_*.py` are not part of any job — run them by hand.

5. **Inspect the built artifacts before pushing a tag.** `twine check` validates that
   the long description renders and that the metadata is well formed:

   ```bash
   python3 -m build
   python3 -m twine check --strict dist/*
   python3 -m venv /tmp/check
   /tmp/check/bin/pip install dist/gradiex-*.whl
   MPLBACKEND=Agg /tmp/check/bin/python -c \
     "import gradiex; print(gradiex.__version__)"
   ```

6. **Land the release on the default branch.** The tag has to point at a commit that
   contains the workflow — if `release.yml` is untracked or unmerged, pushing the tag
   starts no pipeline at all and you will wait for a upload that was never queued.

7. **Tag and push.** The tag is the publish button:

   ```bash
   git tag -a v0.0.3 -m "v0.0.3"
   git push origin main --follow-tags     # or: git push origin v0.0.3
   ```

8. **Watch the run** under the repository's *Actions* tab. `publish` needs
   `wheels` and `sdist` to succeed first; if any wheel fails its XOR test, nothing is
   uploaded.

9. **Verify on PyPI**, then install the published version into a fresh environment and
   train something. A wheel that imports is not the same as a wheel that works — which
   is why CI trains before uploading.

## The build matrix

Defined in `[tool.cibuildwheel]` in `pyproject.toml`, not in the workflow, so the
same configuration applies to a local `cibuildwheel` run:

- **CPython 3.9–3.14.** 3.9 is the floor declared in `[project]`; 3.14 is what
  development happens on.
- **Linux** is built on both `ubuntu-latest` (x86_64) and `ubuntu-24.04-arm`
  (aarch64, native — no QEMU). cibuildwheel selects manylinux *and* musllinux, so
  Alpine gets wheels too.
- **macOS** is built on `macos-13` (Intel) and `macos-14` (Apple silicon).
  `MACOSX_DEPLOYMENT_TARGET` is set per runner. Without it, a wheel is tagged with
  whatever macOS the runner happens to be running (`macosx_26_0`) and installs almost
  nowhere.
- **Windows** is `windows-latest`, x64.
- **32-bit x86 and win32 are skipped**: there are no numpy/matplotlib wheels for them
  to test against.
- **manylinux image is `manylinux_2_28`**, not the 2014 default: `core/` is C++20 and
  needs `<numbers>`, which wants GCC 11+ / Clang 13+; the `manylinux2014` image ships
  GCC 10 and cannot build this.
- `MPLBACKEND=Agg` is set for the test environment so the plotting layer does not try
  to open a window on a headless runner.

## After the release

- `docs/newversion.py <version>` copies the documentation to a new version directory
  and restamps the cross-version navigation. It does **not** touch `pyproject.toml` or
  `python/gradiex_module.cpp`; update those yourself.
- Update the version badge in `README.md`.
- Add a `CHANGELOG.md` section for the next version if the release did not already
  close out that version.
