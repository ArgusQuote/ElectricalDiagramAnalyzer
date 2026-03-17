# How Skills and Rules Connect

Sourced from: [Builder.io Guide](https://builder.io/blog/agent-skills-rules-commands), [Cursor Rules Docs](https://cursor.com/docs/context/rules), [Cursor Skills Docs](https://cursor.com/docs/context/skills), [Atlan Framework](https://blog.atlan.com/engineering/cursor-rules/)

---

## The Fundamental Split

**Rules** = passive, always-in-context behavioral constraints.
- Applied at the start of model context
- Context cost is always paid (for Always Apply) or paid when files match (for globs)
- Think: "What the agent must always know or obey"

**Skills** = active, discoverable capabilities with procedural steps.
- Agent loads only when it determines the skill is relevant
- Context cost paid on demand via progressive disclosure
- Think: "What the agent can do when a specific task requires it"

**Commands** = explicit user-triggered prompts.
- User types `/command-name` to invoke
- Deterministic: you call it, the tool injects the prompt
- Think: "Shortcuts for repeatable actions"

---

## The Builder.io Litmus Test

> "Would you want this instruction to apply even when you are not thinking about it?"

**Yes** --> It is a **Rule** (Always Apply or glob-scoped)
**No** --> It is likely a **Skill** or a **Command**

Further classification:
- Is it a procedure with multiple steps? --> **Skill**
- Is it a one-shot prompt you trigger explicitly? --> **Command**
- Is it optional expertise loaded on demand? --> **Skill**

---

## Decision Tree

```
Is this instruction always needed, regardless of task?
├── YES --> RULE (alwaysApply: true)
│
├── Only when working with specific file types?
│   └── YES --> RULE (with globs: **/*.ts, **/*.py, etc.)
│
├── Is it a multi-step procedure or workflow?
│   └── YES --> SKILL (agent loads when task matches description)
│
├── Is it a one-shot prompt triggered by the user?
│   └── YES --> COMMAND (user types /command-name)
│
└── Is it situational guidance for specific scenarios?
    └── YES --> RULE (Apply Intelligently, alwaysApply: false)
```

---

## Concrete Classification Examples

| Instruction | Type | Reasoning |
|---|---|---|
| "Never commit .env files" | Rule (Always) | Non-negotiable safety, applies to every interaction |
| "Project uses [framework] with [key conventions]" | Rule (Always) | Core project fact the agent always needs |
| "Use dependency injection via constructors" | Rule (Globs: `**/*.ts`, `**/*.py`) | Convention for specific file type |
| "When you touch billing code, run 3 integration tests" | Skill | Conditional procedure, only relevant for billing work |
| "When writing release notes, follow this checklist" | Skill | Multi-step workflow, not needed for every task |
| "How to deploy to staging vs production" | Skill | Complex procedure loaded when deployment is discussed |
| "The design system uses these token names" | Rule (Globs: `**/*.css`, `**/*.tsx`) | Convention for specific files |
| "Generate a changelog entry" | Command | User-triggered, one-shot action |
| "Database migration procedure with Flyway" | Skill | Multi-step workflow, only when creating migrations |
| "All REST endpoints must return proper error responses" | Rule (Globs: `**/api/**`, `**/resource/**`) | Convention that applies to API code |

---

## The Routing Pattern

Rules can serve as lightweight routing logic that points to skills. This keeps the always-on prompt small while making the agent adaptable.

### How It Works

Create a short Always Apply rule that tells the agent when to invoke specific skills:

```markdown
---
description: Routing rules for when to load specialized skills
alwaysApply: true
---

# Skill Routing

- When changing UI components, load the `ui-change` skill
- When debugging production errors, load the `incident-triage` skill
- When creating database migrations, load the `migration-workflow` skill
- When writing release notes, load the `release-notes` skill
```

### Why This Works

- The rule itself costs minimal tokens (just a few routing lines)
- The skill content (potentially hundreds of lines) is loaded only when needed
- The agent gets clear signals about which skill to activate for which task
- You can add new skills without increasing the always-on context cost

---

## Context Cost Comparison

| Approach | Context Cost | When Paid |
|---|---|---|
| Rule (Always Apply) | Every token, every session | Always |
| Rule (Globs) | Every token, when file matches | On file open |
| Rule (Apply Intelligently) | Description tokens always; full content when agent decides | Partial always, full on demand |
| Skill | ~100 tokens for metadata always; full content when activated | Metadata always, rest on demand |

### Implication

If you have a 200-line workflow guide, making it an Always Apply rule costs 200 lines of context in every session. Making it a skill costs ~1 line of metadata in every session, plus 200 lines only when the agent decides it is relevant.

**Rule of thumb**: If it is over 30-40 lines and not needed every session, it should be a skill.

---

## When to Promote or Demote

### Promote Skill to Rule

If the agent frequently fails to load a skill when it should, or if the content is critical enough that missing it causes errors:
1. Extract the key constraints from the skill
2. Create a short Always Apply rule with just those constraints
3. Keep the detailed procedure as a skill
4. Use the routing pattern to connect them

### Demote Rule to Skill

If a rule is long (>50 lines) and only applies to specific workflows:
1. Move the procedural content to a skill
2. Replace the rule with a short routing entry
3. Or convert to Apply Intelligently if it is not workflow-heavy

### When to Split

If a rule or skill tries to cover too many concerns:
- Rules: Split into separate `.mdc` files, one concept per file
- Skills: Split into separate skill directories, or move content to `references/`

---

## Integration Patterns in Practice

### Pattern 1: Rules Define Standards, Skills Define Procedures

**Rule** (Always Apply): "All REST endpoints must handle errors with RFC 7807 Problem Details format."
**Skill** (loaded on demand): "How to implement RFC 7807 error handling" -- with step-by-step code, examples, and validation.

### Pattern 2: Rules Set Boundaries, Skills Provide Playbooks

**Rule** (Always Apply): "Never deploy without running the full test suite."
**Skill** (loaded on demand): "Deployment playbook" -- with staging vs production procedures, rollback steps, and verification checklists.

### Pattern 3: Globs Provide Context, Skills Provide Workflow

**Rule** (Globs: `**/package.json`, `**/pyproject.toml`): "This project uses [package manager] for dependency management. Do not add dependencies without approval."
**Skill** (loaded on demand): "Adding a new dependency" -- with procedure for selecting, adding, configuring, and testing new packages.

---

## Summary: The Right Split

| Characteristic | Rule | Skill |
|---|---|---|
| **Applies to** | Every session or specific files | Specific tasks on demand |
| **Length** | Short (under 80 lines ideal) | Can be longer (under 500 lines) |
| **Content** | Constraints, conventions, facts | Procedures, workflows, playbooks |
| **Trigger** | Automatic (always/globs/agent) | Agent matches task to description |
| **Context cost** | Paid upfront | Paid only when activated |
| **Tone** | Assertive ("Always do X") | Procedural ("Step 1, Step 2...") |
| **Changes** | Rarely (stable conventions) | More often (evolving procedures) |
