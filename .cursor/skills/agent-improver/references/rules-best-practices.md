# Rule Authoring Best Practices

Sourced from: [Cursor Official Rules Docs](https://cursor.com/docs/context/rules), [Atlan Engineering Blog](https://blog.atlan.com/engineering/cursor-rules/), [Builder.io Guide](https://builder.io/blog/agent-skills-rules-commands)

---

## Rule File Format (Cursor Official Docs)

Rules are markdown files in `.cursor/rules/` with YAML frontmatter. Cursor supports both `.md` and `.mdc` extensions. Use `.mdc` for files with frontmatter to specify `description` and `globs`.

```
.cursor/rules/
├── react-patterns.mdc       # Rule with frontmatter
├── api-guidelines.md         # Simple markdown rule
└── frontend/                 # Organize rules in folders
    └── components.md
```

### Rule Anatomy

```markdown
---
description: Brief description of what this rule does
globs: **/*.ts
alwaysApply: false
---

# Rule Title

Rule content here with concrete examples and assertive instructions.
```

### Frontmatter Fields

| Field | Type | Description |
|---|---|---|
| `description` | string | What the rule does (shown in rule picker, used by agent for relevance) |
| `globs` | string | File pattern -- rule applies when matching files are open |
| `alwaysApply` | boolean | If true, applies to every session regardless of context |

---

## The 4 Rule Types (Cursor Docs)

### Always Apply (`alwaysApply: true`)

Applied to every AI interaction. No conditions.

**Use for**: Core project patterns, safety constraints, naming conventions, architecture decisions, project overview.

```yaml
---
description: Core coding standards for the project
alwaysApply: true
---
```

### Apply Intelligently (`alwaysApply: false`, no globs)

Agent decides if the rule is relevant based on its description. The description quality is critical -- if it is vague, the rule may never trigger.

**Use for**: Workflow guidance, situational patterns, conditional processes, documentation the agent consults as needed.

```yaml
---
description: Database migration patterns and rollback procedures. Use when creating or modifying database migrations.
alwaysApply: false
---
```

### Apply to Specific Files (globs set)

Auto-attached when a file matching the glob pattern is open.

**Use for**: Language-specific conventions, file-type patterns.

```yaml
---
description: Coding conventions for API/service layer
globs: **/*.ts
alwaysApply: false
---
```

### Apply Manually

Only applied when explicitly invoked with `@rule-name` in chat.

**Use for**: One-off helpers, debugging tools, experimental workflows, specialized operations.

---

## The Builder.io Litmus Test

> "Would you want this instruction to apply even when you are not thinking about it?"

- **Yes** --> `alwaysApply: true` (it is a core constraint)
- **No** --> Consider globs or Apply Intelligently

### Concrete Examples

| Instruction | Classification | Why |
|---|---|---|
| "Never commit .env files" | Always Apply | Non-negotiable safety constraint |
| "Use 2-space indentation" | Apply to Specific Files (`**/*.ts`, `**/*.py`) | Only relevant for source files |
| "Database migration procedures" | Apply Intelligently | Only when working on migrations |
| "Emergency rollback procedure" | Apply Manually | Rarely needed, explicit invocation |
| "Project uses [framework] with [key patterns]" | Always Apply | Core project fact |
| "How to run the test suite" | Apply Intelligently | Only when agent is running tests |

---

## The Atlan 4-Category Framework

A proven real-world taxonomy for organizing rules:

### 1. Project Rules (Always Apply)

What this project IS. Structure, architecture, overview, glossary.

Like handing the agent your README, onboarding doc, and architecture guide in a form it understands.

**Without this**: Agent is flying blind, does not know folder layout, sacred patterns, or project intent.

**Examples**: Project overview, folder structure map, git branching strategy, change propagation rules, glossary of domain terms.

### 2. Tech Stack Rules (Always or Agent Requested)

HOW your team writes code. Language conventions, framework patterns, API standards.

**Examples**: Naming conventions, REST endpoint patterns, dependency injection rules, database column naming, test framework usage.

**Rule Type**: Use Always for core languages. Use Agent Requested for specific tools or frameworks the agent may not always need.

### 3. Micro Workflow Rules (Agent Requested or Manual)

HOW your team behaves. Repeated, often subconscious patterns.

**Examples**: Logging structure, auth wrappers, feature flag conventions, migration patterns, event tracking, error handling flows.

**Impact**: These compound fast across features and teammates. Getting them right in rules means fewer reminders and fewer mistakes.

### 4. Meta Rules (Always Apply)

Rules about your rules. How the agent should interpret, prioritize, and apply all other rules.

**Examples**: Rule folder organization, conflict resolution between rules, when to consult which folder, context management instructions.

**Without this**: Once you have multiple rules across different contexts, ambiguity creates confusion. Meta rules create coherence.

---

## Writing Philosophy (Atlan)

### Be Assertive, Not Suggestive

LLMs do not handle nuance well. Vague instructions lead to unpredictable behavior.

**Bad**: "Consider using TypeScript interfaces"
**Good**: "Always define TypeScript interfaces for component props"

Use action words. Be direct. Be unmissable.

### Place Important Context Last

LLMs often prioritize information at the end of a rule file. Structure your rules:

1. What to do
2. How to do it (with examples)
3. Why it matters (most important constraints here)

### One Concept Per Rule

Do not combine unrelated concerns in a single file. Create separate rule files for different topics.

**Bad**: One file covering naming conventions, error handling, logging, AND database patterns.
**Good**: Separate files for each concern.

### Target Line Count

- **Ideal**: 50-80 lines per rule
- **Maximum**: 500 lines (per Cursor docs)
- If approaching the limit, split into multiple focused rules

---

## Cursor Official Best Practices

### DO

- Keep rules under 500 lines
- Split large rules into multiple composable rules
- Provide concrete examples or reference files
- Write rules like clear internal docs
- Reuse rules when repeating prompts in chat
- Reference files instead of copying their contents
- Check rules into git for team sharing
- Update rules when the agent makes mistakes

### DO NOT

- Copy entire style guides (use a linter instead -- agent already knows common conventions)
- Document every possible command (agent knows common tools like npm, git, pytest)
- Add instructions for edge cases that rarely apply
- Duplicate what is already in your codebase (point to canonical examples instead)
- Over-optimize before understanding your patterns

**Start simple**: Add rules only when you notice the agent making the same mistake repeatedly.

---

## Rule Precedence (Cursor Docs)

Rules are applied in this order, with earlier sources taking precedence on conflicts:

1. **Team Rules** (from Cursor dashboard -- Team/Enterprise plans)
2. **Project Rules** (`.cursor/rules/`)
3. **User Rules** (global preferences in Cursor Settings)

---

## Debugging Rules

If a rule is not being applied:

1. **Check description quality** (for Apply Intelligently): Vague descriptions cause the agent to skip the rule
2. **Verify glob patterns** match actual file paths
3. **Test with `@rule-name`** in chat to force-apply and verify the content works
4. **Check for conflicts** between rules that may contradict each other
5. **Verify file location**: Must be in `.cursor/rules/` directory

### Known Issue

"Apply Intelligently" rules sometimes appear not to apply (reported on Cursor forum). The LLM's base behavior can override contextual rules. If reliability is required, promote the rule to Always Apply.

---

## Complete Rule Examples

### Always Apply -- Project Overview

```markdown
---
description: Core project overview and architecture
alwaysApply: true
---

# Project Overview

<!-- Describe your project type, framework, and key technologies -->

## Key Conventions
- All REST/API handlers in `src/.../api/` (or equivalent)
- All business logic in `src/.../service/`
- All domain entities in `src/.../model/`
- Configuration in project config files

## Build and Run
<!-- Add your project's dev, test, and build commands -->
```

### Apply to Specific Files -- API Conventions

```markdown
---
description: API and service layer coding conventions for this project
globs: **/*.ts
alwaysApply: false
---

# API Conventions

- Use constructor/dependency injection, not global singletons
- Follow REST resource conventions (proper HTTP methods, status codes)
- Return typed responses with appropriate error handling
- Keep data access logic separate from business logic
- Use transactions for methods that write to the database
```

### Apply Intelligently -- Migration Workflow

```markdown
---
description: Database migration workflow using Flyway. Use when creating or modifying database migrations, schema changes, or Flyway scripts.
alwaysApply: false
---

# Database Migration Workflow

1. Create migration file in the project's migration directory
2. Name format: `V{version}__{description}.sql` (or your project's convention)
3. Always include a rollback comment at the top
4. Test migration using your project's migration tool
5. Verify migrations were applied successfully

Never modify an existing migration file that has been applied.
```
