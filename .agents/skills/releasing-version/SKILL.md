---
name: releasing-version
description: "Release a new version of NNLang. Use when asked to release, deploy, cut a release, bump version, or create a new version tag."
---

# Releasing a New Version

Follow these steps in order. Stop and ask the user at each decision point marked with 🔲.

## Step 1 — Pre-flight Checks

### 1a. Verify documentation is up to date

- Read `CHANGELOG.md` and confirm the `[Unreleased]` section is non-empty and accurately describes the changes since the last release.
- Read `README.md` and check the "Current Scope" section version reference matches (or will match) the new version.
- If anything is missing or stale, fix it before proceeding.

### 1b. Generate release notes

Run the release notes generator:

```bash
./scripts/generate-release-notes.sh
```

**Do NOT run this yet** — it strips `[Unreleased]`. It will be run after the CHANGELOG is finalized in Step 3.

### 1c. Run tests

```bash
cargo test
```

If tests fail, stop and report the failures.

## Step 2 — Determine Release Type

Analyze the `[Unreleased]` section of `CHANGELOG.md` and classify the release:

| Type | Criteria | Version bump |
|------|----------|-------------|
| **Major** | Breaking changes: removed features, changed CLI flags, changed file formats, incompatible API changes | `X+1.0.0` |
| **Minor** | New features, new layers, new CLI commands, new config options (backward-compatible) | `X.Y+1.0` |
| **Patch** | Bug fixes, documentation fixes, performance improvements (no new features) | `X.Y.Z+1` |

Rules for classification:
- If `### Added` contains new user-facing features → **minor** (at minimum)
- If `### Fixed` only → **patch**
- If `### Changed` or `### Removed` contains breaking changes → **major**
- When in doubt, prefer minor over patch

Read the current version from `Cargo.toml` (`version = "X.Y.Z"`).

🔲 **Ask the user:** "Based on the changes, this looks like a **[type]** release: `vCURRENT` → `vNEW`. Proceed?"

## Step 3 — Finalize CHANGELOG

1. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] — YYYY-MM-DD` using today's date.
2. Add a fresh empty `## [Unreleased]` section above it.
3. Now generate the release notes:

```bash
./scripts/generate-release-notes.sh
```

4. Verify the generated `docs/src/release-notes.md` contains the new version and does NOT contain `[Unreleased]`.

## Step 4 — Bump Version

Update `Cargo.toml`:

```
version = "X.Y.Z"
```

Run `cargo check` to update `Cargo.lock`.

Update the README badge URL if it references a specific release tag:

```
grep -n 'v[0-9]' README.md
```

Fix any hardcoded version references.

## Step 5 — Commit and Push

Stage and commit all release changes:

```bash
git add Cargo.toml Cargo.lock CHANGELOG.md README.md docs/src/release-notes.md
git commit -m "release: vX.Y.Z"
```

Check if the commit has been pushed:

```bash
git status -sb  # shows ahead/behind
```

🔲 If not pushed: **Ask the user:** "Local commits need to be pushed before tagging. Push to origin/main now?"

If yes:

```bash
git push origin main
```

## Step 6 — Verify CI

Check if CI is passing for the pushed commit:

```bash
gh run list --branch main --limit 5
```

If `gh` is not available, provide the URL:

```
https://github.com/gdesouza/nnl/actions?query=branch%3Amain
```

🔲 If CI is failing or pending: **Ask the user:** "CI is [status] for this commit. Create the release tag anyway?"

## Step 7 — Tag and Push

Create and push the release tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

This triggers the `release.yml` workflow which builds release artifacts.

Report:

```
✅ Released vX.Y.Z
   Tag: vX.Y.Z
   Release workflow: https://github.com/gdesouza/nnl/actions/workflows/release.yml
```

## Quick Reference

```
Pre-flight:  cargo test + check CHANGELOG + check README
Classify:    patch / minor / major based on [Unreleased] content
Finalize:    freeze CHANGELOG, generate release notes
Bump:        Cargo.toml version + cargo check
Commit:      git commit -m "release: vX.Y.Z"
Push:        git push origin main
Verify CI:   gh run list --branch main
Tag:         git tag vX.Y.Z && git push origin vX.Y.Z
```
