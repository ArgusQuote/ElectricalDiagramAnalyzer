---
name: reviewing-code
description: Reviews code for correctness, maintainability, and adherence to project standards using a structured checklist with severity levels. Use when reviewing pull requests, examining code changes, or when the user asks for a code review.
---

# Reviewing Code

Perform a structured code review that catches bugs, enforces standards, and provides actionable feedback organized by severity.

## Inputs

- Code diff or file(s) to review
- Project coding standards (from `.cursor/rules/`)
- Context about the purpose of the changes (PR description, issue reference, or user explanation)

## Steps

Copy this checklist and track progress:

```
Task Progress:
- [ ] Step 1: Understand the context
- [ ] Step 2: Read project standards
- [ ] Step 3: Check correctness
- [ ] Step 4: Check maintainability
- [ ] Step 5: Check test coverage
- [ ] Step 6: Check convention adherence
- [ ] Step 7: Provide structured feedback
```

### Step 1: Understand the context

Before reviewing code, understand what the change is trying to accomplish:
- Read the PR description, commit message, or related issue
- Ask the user for context if the purpose is unclear
- Identify the scope: which files changed and why

### Step 2: Read project standards

Load relevant project rules to review against:
- Coding standards from `.cursor/rules/generic/coding-standards.mdc`
- Domain-specific conventions from `.cursor/rules/domain/` (if applicable)
- Security practices from `.cursor/rules/generic/security-practices.mdc`

### Step 3: Check correctness

Review for functional correctness:
- Logic errors and incorrect conditions
- Edge cases not handled (null, empty, boundary values)
- Off-by-one errors in loops or ranges
- Race conditions or concurrency issues
- Resource leaks (unclosed connections, file handles, streams)
- Error handling gaps (unhandled exceptions, missing error cases)

### Step 4: Check maintainability

Review for long-term code health:
- Naming clarity: do names reveal intent?
- Complexity: are methods focused on a single responsibility?
- Duplication: is logic repeated that should be extracted?
- Readability: can a new developer understand this without explanation?
- Magic numbers or strings that should be named constants

### Step 5: Check test coverage

Review whether changes are adequately tested:
- Are new code paths covered by tests?
- Are edge cases tested?
- Are error paths tested?
- If tests are missing, specify what tests should be added

### Step 6: Check convention adherence

Verify changes follow project conventions:
- File organization matches project structure
- Naming follows project patterns
- Error handling follows established patterns
- Import/dependency patterns are consistent

### Step 7: Provide structured feedback

Organize all findings by severity:

**Critical** -- Must fix before merge:
- Bugs, logic errors, security vulnerabilities
- Data loss or corruption risks
- Missing error handling on critical paths

**Suggestion** -- Should improve:
- Readability and naming improvements
- Performance concerns
- Missing tests for important paths
- Pattern inconsistencies

**Nice-to-have** -- Optional enhancement:
- Minor style preferences
- Additional documentation
- Optimization opportunities that are not urgent

For each finding, provide:
1. The specific file and line (or line range)
2. What the issue is
3. Why it matters
4. A concrete suggestion for how to fix it

## Checks

- All Critical findings are addressed before approving
- Feedback is specific and actionable (not vague observations)
- Each finding includes a concrete fix suggestion

## Stop Conditions

- If the review scope exceeds 500 lines of changes, suggest breaking the PR into smaller pieces for effective review
- If changes touch unfamiliar domain logic, ask the user for context before reviewing correctness
- If the code is generated or auto-formatted, focus review on the source/configuration rather than the output

## Recovery

- If the author disagrees with a finding, discuss the tradeoff and document the decision
- If a Critical finding reveals a deeper architectural issue, flag it separately and suggest a follow-up task rather than blocking the current PR
