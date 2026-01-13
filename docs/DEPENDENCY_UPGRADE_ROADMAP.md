# Dependency Upgrade Roadmap (mistral.rs)

This document outlines a safe, CI-parity approach for upgrading the dependencies that currently have newer compatible releases available.

## Scope

From the current `cargo update --verbose` output, the following crates have newer versions available but were not selected by Cargo’s resolver:

- `cfgrammar` (available `v0.14.0`)
- `generic-array` (available `v0.14.9`)
- `hashbrown` (available `v0.16.1`)
- `html2text` (available `v0.16.5`)
- `indicatif` (available `v0.18.3`)
- `lrtable` (available `v0.14.0`)
- `matchit` (available `v0.8.6`)
- `oci-spec` (available `v0.6.8`)
- `pyo3` (available `v0.27.2`)
- `pyo3-build-config` (available `v0.27.2`)
- `radix_trie` (available `v0.3.0`)
- `rust-mcp-schema` (available `v0.9.1`)
- `rust-mcp-sdk` (available `v0.7.4`)
- `rustyline` (available `v17.0.2`)
- `safetensors` (available `v0.7.0`)
- `schemars` (available `v1.2.0`)
- `scraper` (available `v0.25.0`)
- `sysinfo` (available `v0.37.2`)
- `tokenizers` (available `v0.22.2`)
- `tokio-tungstenite` (available `v0.28.0`)
- `toml` (available `v0.9.10+spec-1.1.0`)
- `tqdm` (available `v0.8.0`)
- `yoke` (available `v0.8.1`)

## Goals

- Upgrade incrementally without breaking core inference / server / Python bindings.
- Keep changes reviewable (small PRs, tight scopes).
- Maintain CI parity (fmt/clippy/tests/docs).

## Non-goals

- Large refactors unrelated to dependency uplift.
- Changing public APIs unless required by an upgrade.

## Risks by dependency

### High risk

- `pyo3`, `pyo3-build-config`
  - Python bindings tend to be sensitive to breaking API changes.
  - May require Rust MSRV adjustments, feature flag changes, or PyO3 macro changes.

- `tokenizers`
  - Often tightly coupled with `hf-hub`, `serde`, and on-disk tokenizer formats.
  - Potential for subtle runtime regressions.

- `schemars` (major bump `0.x` -> `1.x`)
  - Likely API changes.
  - Anything generating JSON schema or OpenAPI can break.

- `toml` (major bump)
  - Parser/serializer behavior can change.
  - Can surface as config parsing differences.

### Medium risk

- `safetensors`
  - Impacts weight loading; even “minor” incompatibilities can be costly.

- `sysinfo`
  - Impacts runtime hardware introspection.

- `rust-mcp-schema`, `rust-mcp-sdk`
  - Impacts MCP integration; API churn is common.

- `tokio-tungstenite`
  - WebSocket behavior changes; can break server/web chat in subtle ways.

### Low risk

- `generic-array`, `hashbrown`, `yoke`
- `matchit`, `oci-spec`
- `indicatif`, `tqdm`
- `scraper`, `html2text`
- `radix_trie`
- `cfgrammar`, `lrtable`
- `rustyline`

## Recommended upgrade sequence

### Phase 0: Baseline snapshot (no dependency changes)

- Record the current known-good state (commit SHA).
- Confirm current CI parity locally:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --tests --examples -- -D warnings`
  - `cargo test --workspace`
  - `cargo doc --workspace`

### Phase 1: Low-risk patch/minor upgrades

Goal: quick wins to reduce overall drift.

- Target: `generic-array`, `hashbrown`, `yoke`, `matchit`, `oci-spec`, `indicatif`, `tqdm`, `radix_trie`, `rustyline`, `scraper`, `html2text`.

Approach:
- Use targeted updates, not a blanket update.
- Prefer `cargo update -p crate_name` where possible.

Validation:
- `cargo check --workspace`
- `cargo test --workspace` (or at least `-p mistralrs-core -p mistralrs-server-core`)

### Phase 2: Medium-risk runtime-impacting upgrades

- Target: `safetensors`, `sysinfo`, `tokio-tungstenite`.

Additional validation:
- Run a minimal smoke test:
  - Start `mistralrs-server` and hit `/v1/models` and a simple `/v1/chat/completions` request.
- Run one local model load path that exercises:
  - tokenization
  - safetensors load
  - a short inference

### Phase 3: Schema/config surface upgrades

- Target: `toml`, `schemars`.

Approach:
- Make these isolated PRs.
- Expect compile errors (API changes) and config behavior changes.

Validation:
- Run config parsing tests (or add a minimal test that parses representative config files).
- If schema output is used externally (OpenAPI/MCP), snapshot and diff outputs.

### Phase 4: High-risk bindings and model IO upgrades

- Target: `pyo3`, `pyo3-build-config`, `tokenizers`.

Approach:
- Upgrade PyO3 as its own PR.
- Upgrade tokenizers as its own PR.

Validation:
- Build the Python bindings crate (`mistralrs-pyo3`) in CI parity.
- Run a minimal Python import + trivial call smoke test (if you have one).

## Guardrails / workflow

- One PR per phase (or per major dependency in phases 3/4).
- Keep `Cargo.lock` changes scoped and explain them.
- If the upgrade causes API churn, add shims rather than scattering version-conditional code.
- If MSRV changes are required, document the change explicitly and bump in a dedicated commit.

## Rollback strategy

- If a dependency causes regressions:
  - revert the PR
  - or pin via `Cargo.toml`/`[patch.crates-io]` temporarily
  - file an issue tracking the required follow-up changes

## Suggested deliverables

- PR 1: Phase 1 low-risk uplift
- PR 2: Phase 2 runtime-impacting uplift + smoke test notes
- PR 3: `toml` uplift
- PR 4: `schemars` uplift
- PR 5: `pyo3` uplift
- PR 6: `tokenizers` uplift
