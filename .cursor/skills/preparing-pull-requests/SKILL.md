---
name: preparing-pull-requests
description: Prepares high-quality commits and pull requests by analyzing changes, generating descriptive messages, and enforcing a review checklist. Use when the user asks to commit, create a PR, push changes, or prepare work for review.
---

# Preparing Pull Requests

Create well-structured commits and pull requests that are easy to review and maintain a clean project history.

## Inputs

- Staged or unstaged git changes
- Access to git log for commit message style conventions
- Project's git conventions (from `.cursor/rules/generic/git-conventions.mdc`)

## Steps

Copy this checklist and track progress:

```
Task Progress:
- [ ] Step 1: Understand all changes
- [ ] Step 2: Check for secrets
- [ ] Step 3: Verify changes work
- [ ] Step 4: Classify the change
- [ ] Step 5: Draft commit message
- [ ] Step 6: Stage and commit
- [ ] Step 7: Create PR (if requested)
```

### Step 1: Understand all changes

Run these commands to understand the full scope of changes:
- `git status` -- see all modified, added, and deleted files
- `git diff` -- see unstaged changes in detail
- `git diff --staged` -- see already staged changes
- `git log --oneline -10` -- understand recent commit message style

### Step 2: Check for secrets

Before committing, verify no sensitive data is included:
- Scan for files that commonly contain secrets: .env, credentials.json, *.pem, *.key
- Check diff content for patterns: API keys, tokens, passwords, connection strings
- If potential secrets are found, **stop and warn the user** before proceeding

### Step 3: Verify changes work

Load the `verifying-changes` skill and run verification:
- Run tests related to changed files
- Run linter on changed files
- Confirm no regressions introduced

### Step 4: Classify the change

Determine the type of change for the commit message:
- **feature**: New functionality added
- **fix**: Bug fix
- **refactor**: Code restructuring without behavior change
- **docs**: Documentation only
- **test**: Adding or updating tests
- **chore**: Build, config, or tooling changes

### Step 5: Draft commit message

Write a commit message following project conventions:
- **Summary line**: Imperative mood, under 50 characters, describes the WHY
- **Body** (if needed): Explain context, motivation, and trade-offs
- Reference issue numbers when applicable
- Match the style observed in `git log`

### Step 6: Stage and commit

- Stage only the files relevant to this logical change
- Do not mix unrelated changes in a single commit
- Create the commit with the drafted message

### Step 7: Create PR (if requested)

If the user asks for a pull request:
- Generate a PR title summarizing the change
- Write a PR body with:
  - **Summary**: 1-3 bullet points describing what changed and why
  - **Test plan**: How the changes were verified
  - **Checklist**: Applicable items checked off
- Push the branch and create the PR

## Checks

- Commit message follows project conventions (imperative mood, concise summary)
- No secrets or credentials in staged files
- All verification checks pass before committing
- PR description is clear and includes a test plan

## Stop Conditions

- If staged files contain potential secrets (.env, credentials, API keys), stop and warn the user immediately
- If there are merge conflicts, stop and ask for resolution guidance
- If changes span multiple unrelated concerns, suggest splitting into separate commits

## Recovery

- If CI fails after push, load the `debugging-issues` skill to investigate the failure
- If the commit message does not match project style, amend the commit before pushing (only if not yet pushed)
