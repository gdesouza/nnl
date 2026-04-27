# AGENTS

## Scope
- Single Rust crate, not a workspace. The only binary is `nnc` (`src/main.rs`), and `driver::run` in `src/driver.rs` is the real command entrypoint.
- Core pipeline is `src/syntax/` -> `src/sema/` -> `src/ir/` -> `src/weights/` -> `src/codegen/`. Most feature work spans several of these modules.

## Commands
- Match CI before finishing: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test`.
- Focused tests: `cargo test --test compile <name>`, `cargo test --test sema <name>`, `cargo test --test weights <name>`, or `cargo test <unit_test_name>` for in-file unit tests.
- Docs build is mdBook-based: install with `cargo install --locked mdbook`, then run `mdbook build` in `docs/`.
- Security checks in CI are separate from normal test runs: `cargo audit` and `cargo deny check`.

## Repo-Specific Gotchas
- `nnc compile` shells out to the host C toolchain. `exe`/`obj`/`shared` use `cc`; `lib` also requires `ar`; `--target-triple <triple>` switches to `<triple>-gcc`.
- `nnc test` compiles a temporary executable and compares `.npy` input/output by feeding raw float32 bytes over stdin/stdout. Compile/integration tests depend on a working C compiler, not just Rust.
- Parser snapshot tests live in `src/syntax/parser.rs` with snapshots in `src/syntax/snapshots/`. If AST/debug output changes, update the snapshots too.
- Weight-loading behavior is important design surface here: docs explicitly prioritize weights directory of `.npy` files, then `.npz`, then ONNX initializers (`docs/src/DESIGN.md`).

## Files Worth Checking First
- `README.md` for supported CLI flows and current feature scope.
- `docs/src/cli.md` and `docs/src/codegen.md` for exact `nnc` behavior and emit-mode expectations.
- `tests/compile.rs` for end-to-end expectations; it is the best source for how generated binaries are supposed to behave.

## Repo Skills
- Load the `releasing-version` skill before doing any release/tag/version-bump work. It encodes the repo's release order, decision points, and changelog/release-notes flow.
- Load the `using-serena-mcp` skill when doing symbol-level code navigation. It is specifically meant for repo exploration and tracing definitions/references.

## Noise To Ignore
- Do not treat `target/` or `docs/book/` as source; both are generated and ignored.
- `examples/import_test/` contains generated artifacts that are also ignored.
