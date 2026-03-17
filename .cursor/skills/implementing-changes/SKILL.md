---
name: implementing-changes
description: Guides the agent through a complete implementation workflow -- understand requirements, plan the approach, implement incrementally, verify after each change, and self-assess quality before presenting results. Use when building features, implementing fixes, writing new code, or making any substantive code changes.
---

# Implementing Changes

Follow the plan-implement-verify loop to build correct code. Every increment is verified before moving on. Results are self-assessed before presenting to the user.

## Inputs

- User requirements (feature request, bug report, task description)
- Relevant codebase context (files to modify, existing patterns, project rules)
- Project's verification commands (test runner, linter, type checker)

## Steps

Copy this checklist and track progress:

```
Task Progress:
- [ ] Step 1: Understand the requirements
- [ ] Step 2: Plan the approach
- [ ] Step 3: Implement an increment
- [ ] Step 4: Verify the increment
- [ ] Step 5: Repeat or continue
- [ ] Step 6: Self-assess quality
- [ ] Step 7: Present results
```

### Step 1: Understand the requirements

Before writing any code:
- Restate what the user is asking for in your own words
- Identify the acceptance criteria -- what does "done" look like?
- Identify which files need to change
- Identify constraints (backwards compatibility, performance, style conventions)

If requirements are ambiguous, ask the user for clarification before proceeding.

### Step 2: Plan the approach

Design the solution before coding:
- Outline the changes needed (which files, which functions, what logic)
- Identify risks or edge cases upfront
- Determine the order of changes (dependencies first, consumers second)
- Decide how to break the work into small, verifiable increments

Each increment should be independently verifiable -- it should not leave the codebase in a broken state.

### Step 3: Implement an increment

Make a small, focused change:
- Modify one file or one logical unit at a time
- Follow the project's coding standards (from `.cursor/rules/`)
- Write or update tests alongside the implementation
- Keep changes minimal -- do not refactor unrelated code in the same increment

### Step 4: Verify the increment

After each increment, run verification immediately. Load the `verifying-changes` skill for the full procedure:

1. **Run tests** on changed files -- all must pass
2. **Run linter** on changed files -- no new errors
3. **Run type checker** if applicable -- no new errors

**If verification passes**: Mark the increment complete and continue to the next one.

**If verification fails**:
- Analyze the failure and identify the root cause
- If the fix is straightforward, apply it and re-verify
- If the failure is complex, load the `debugging-issues` skill for structured diagnosis
- If the same check fails 3 times, stop and present findings to the user

### Step 5: Repeat or continue

Return to Step 3 for the next increment until all planned changes are complete.

After all increments:
- Run the full test suite (not just changed files) to catch regressions
- Confirm all verification checks pass on the complete change set

### Step 6: Self-assess quality

Before presenting results, review your own work. Load the `reviewing-code` skill and apply it to your changes:

**Correctness check**:
- Does the implementation match what the user requested?
- Are all acceptance criteria met?
- Are edge cases handled (null, empty, boundary values, error paths)?
- Is error handling complete and specific?

**Quality check**:
- Are names clear and descriptive?
- Are methods focused on a single responsibility?
- Is there any duplicated logic that should be extracted?
- Are magic numbers or strings replaced with named constants?

**Completeness check**:
- Are tests included for the new or changed code?
- Are comments and documentation updated where needed?
- Were any files left in a modified but incomplete state?

**Uncertainty disclosure**:
- Is there anything you are unsure about in the implementation?
- Are there trade-offs the user should be aware of?
- Are there follow-up tasks or improvements worth mentioning?

If the self-assessment reveals issues, fix them and re-verify before proceeding.

### Step 7: Present results

Present the completed work to the user with a quality summary:

1. **What was done**: Brief description of the changes made
2. **Files changed**: List of modified, added, or deleted files
3. **Verification status**: Confirmation that tests, lint, and type checks pass
4. **Edge cases handled**: Notable edge cases or error paths addressed
5. **Uncertainties** (if any): Anything you are unsure about or that needs user input
6. **Suggested follow-ups** (if any): Improvements or related work worth considering

## Checks

- All tests pass (no failures, no errors)
- Linter reports no errors on changed files
- Type checker reports no errors on changed files
- Implementation matches the user's stated requirements
- Self-assessment reveals no unaddressed issues

## Stop Conditions

- If requirements are ambiguous or contradictory, ask the user for clarification before implementing
- If verification fails 3 times on the same issue, stop and present findings
- If the change requires modifying unfamiliar domain logic, ask the user for context
- If the planned approach turns out to be wrong mid-implementation, stop, revert, re-plan, and explain the revised approach to the user

## Recovery

- If a verification failure is caused by the current change, load `debugging-issues` to diagnose and fix
- If a verification failure is pre-existing (not caused by current changes), document it separately and do not block on it
- If the planned approach is fundamentally flawed, revert all changes, re-plan with the new understanding, and start again from Step 3
- If you cannot resolve an issue after exhausting debugging, present all findings honestly and let the user decide the path forward
