---
name: debugging-issues
description: Debugs issues using a hypothesis-driven approach -- reproduce, collect evidence, hypothesize, fix, and verify. Use when encountering bugs, error messages, test failures, or when the user asks to fix an issue or investigate unexpected behavior.
---

# Debugging Issues

Apply structured, hypothesis-driven debugging to find and fix the root cause of an issue rather than making random changes.

## Inputs

- Error message, stack trace, bug report, or description of unexpected behavior
- Context about when the issue occurs (always, intermittently, after a specific action)

## Steps

Copy this checklist and track progress:

```
Task Progress:
- [ ] Step 1: Reproduce the issue
- [ ] Step 2: Collect evidence
- [ ] Step 3: Form a hypothesis
- [ ] Step 4: Test the hypothesis
- [ ] Step 5: Verify the fix
- [ ] Step 6: Document the root cause
```

### Step 1: Reproduce the issue

Confirm the issue is reproducible:
- Run the failing test, command, or workflow that triggers the error
- If the user provided an error message, find the code path that produces it
- If the issue is intermittent, identify the conditions that trigger it

If the issue cannot be reproduced, stop and ask the user for more context.

### Step 2: Collect evidence

Gather all relevant information:
- Read the full error message and stack trace
- Identify the exact file, line, and function where the error originates
- Check recent changes (git log, git diff) that may have introduced the issue
- Read relevant log output
- Identify the expected vs actual behavior

### Step 3: Form a hypothesis

State a specific, testable hypothesis about the root cause:
- **Good**: "The NullPointerException occurs because `user.profile` is not loaded when the session expires"
- **Bad**: "Something is wrong with the user module"

The hypothesis must be specific enough to guide a targeted fix.

### Step 4: Test the hypothesis

Make the minimal change needed to test the hypothesis:
- If the hypothesis is about a missing null check, add the check
- If the hypothesis is about incorrect logic, fix the specific condition
- Avoid changing unrelated code while testing a hypothesis

### Step 5: Verify the fix

Confirm the fix resolves the issue without introducing regressions:
- Run the specific failing test or reproduction step -- it must now pass
- Run related tests to check for regressions
- If the `verifying-changes` skill is available, use it for a thorough check

### Step 6: Document the root cause

Briefly record:
- What the root cause was
- Why it happened (not just what was changed)
- What was done to fix it

## Checks

- The original error no longer occurs
- All previously passing tests still pass
- The fix addresses the root cause, not a symptom

## Stop Conditions

- If the bug cannot be reproduced after reasonable effort, stop and ask the user for more context
- If the root cause is in a third-party dependency, stop and inform the user -- recommend updating the dependency or filing an upstream issue
- If the issue requires understanding business logic that is not documented, ask the user to explain the expected behavior

## Recovery

- If a hypothesis is wrong, revert the change, form a new hypothesis based on what was learned, and try again
- After 3 failed hypotheses, stop and present all findings to the user:
  - What was tried
  - What was ruled out
  - What evidence remains unexplained
- Let the user provide additional context before continuing
