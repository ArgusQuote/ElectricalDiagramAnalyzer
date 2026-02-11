---
name: verifying-changes
description: Systematically validates code changes by running tests, linters, and type checkers, then maps failures to specific code locations with remediation steps. Use after implementing changes, before committing, or when the user asks to verify or validate work.
---

# Verifying Changes

Prove that code changes work correctly by running the project's verification tools and fixing any issues found.

## Inputs

- Knowledge of the project's test, lint, and type-check commands (from config files, README, or CLAUDE.md)
- The set of files that were changed (from git diff or recent edits)

## Steps

Copy this checklist and track progress:

```
Task Progress:
- [ ] Step 1: Identify verification commands
- [ ] Step 2: Run tests
- [ ] Step 3: Run linter
- [ ] Step 4: Run type checker
- [ ] Step 5: Map failures to code
- [ ] Step 6: Fix and re-verify
```

### Step 1: Identify verification commands

Determine the project's verification tools by reading config files:
- **Test runner**: pytest, jest, vitest, cargo test, go test, etc.
- **Linter**: ruff, eslint, pylint, clippy, golangci-lint, etc.
- **Type checker**: mypy, pyright, tsc, etc.

If commands are unknown, check package.json scripts, Makefile targets, pyproject.toml sections, or ask the user.

### Step 2: Run tests

Prefer running only tests related to changed files for speed:
- If possible, run tests for the specific changed files first
- If that passes, run the broader test suite to catch regressions
- Capture the full output including any failure details

### Step 3: Run linter

Run the project's linter on changed files:
- Capture all warnings and errors
- Distinguish between new issues (introduced by changes) and pre-existing issues

### Step 4: Run type checker

If the project uses static type checking:
- Run the type checker on the affected files or the full project
- Capture any type errors introduced by the changes

### Step 5: Map failures to code

For each failure found:
- Identify the specific file and line number
- Classify as: test failure, lint error, or type error
- Provide a brief explanation of the root cause
- Suggest a concrete remediation

### Step 6: Fix and re-verify

- Fix each issue, addressing root causes not symptoms
- Re-run the failing check after each fix to confirm resolution
- Repeat until all checks pass

## Checks

All verification commands exit with success (exit code 0). Specifically:
- All tests pass (no failures, no errors)
- Linter reports no errors on changed files
- Type checker reports no errors on changed files

## Stop Conditions

- If the test suite requires external services (database, API, network) that are not available, stop and inform the user
- If more than 10 unrelated test failures appear, ask whether the test suite was passing before the changes
- If the project has no test, lint, or type-check tooling configured, inform the user and suggest setting them up

## Recovery

- If a check fails, analyze the error, fix the root cause, and re-run
- If the same check fails 3 times after attempted fixes, stop and present all findings to the user for guidance
- If a pre-existing failure is discovered (not caused by current changes), note it separately and do not block on it
