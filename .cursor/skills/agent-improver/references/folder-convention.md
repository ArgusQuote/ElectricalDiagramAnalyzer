# Rules Folder Structure Convention

This document defines the standard folder structure for `.cursor/rules/` used by the agent-improver skill's Project Onboard mode.

---

## Directory Layout

```
.cursor/rules/
├── domain/                          # Framework and technology-specific rules
│   ├── api-conventions.mdc
│   ├── dependency-injection.mdc
│   └── framework-config.mdc
│
├── generic/                         # Universal rules for any project
│   ├── cursor-conventions.mdc       # Meta rule: how rules are organized
│   ├── coding-standards.mdc
│   ├── git-conventions.mdc
│   └── security-practices.mdc
│
└── project/                         # Rules specific to THIS project
    ├── team-agreements.mdc
    └── docs/                        # Living documentation rules
        ├── api-patterns.mdc         # External service connection patterns
        ├── architecture.mdc         # System architecture and pipelines
        ├── data-structures.mdc      # Data model documentation
        └── file-locations.mdc       # Code file map and descriptions
```

---

## Folder Descriptions

### domain/

**What belongs here**: Framework-specific and technology-specific rules. These are tied to your tech stack and would change if you switched technologies.

**Examples**:
- REST endpoint patterns for your framework
- Dependency injection conventions
- ORM/entity patterns
- React component conventions
- Database-specific SQL patterns
- Docker/container build patterns

**Key characteristic**: If you migrated frameworks (e.g., Express to Fastify, React to Vue), you would replace these rules. They are about the technology, not the project.

**Typical rule types**: Apply to Specific Files (globs matching framework file types) or Always Apply (if the framework is pervasive).

### generic/

**What belongs here**: Universal rules that apply regardless of programming language, framework, or project. These are transferable across any codebase.

**Examples**:
- Git branching and commit message conventions
- Code documentation standards
- Security practices (no secrets in code, dependency scanning)
- General coding standards (naming, error handling, no magic numbers)
- Meta rules about how the rule system itself works
- Cursor-specific conventions

**Key characteristic**: You could copy these to any project and they would still make sense. They are not tied to any technology or business domain.

**Typical rule types**: Always Apply (for universal standards) or Apply Intelligently (for process-specific guidance).

### project/

**What belongs here**: Rules specific to THIS project that do not fit into domain or generic. Business logic conventions, team-specific agreements, project architecture decisions.

**Examples**:
- Project-specific naming conventions for business entities
- Team workflow agreements
- Deployment environment specifics
- Integration patterns unique to this project's architecture

**Key characteristic**: These rules would not make sense in a different project, even one using the same tech stack.

**Typical rule types**: Always Apply (for core project facts) or Apply Intelligently (for specific workflows).

### project/docs/

**What belongs here**: Living documentation rules that help the agent understand the project's context. These are consulted by the agent when it needs to understand how the project works, not enforced as constraints.

**The 4 standard documentation rules**:

| File | Purpose | Updated When |
|---|---|---|
| `api-patterns.mdc` | External API integrations, connection patterns, auth flows | New API integration added or changed |
| `architecture.mdc` | System architecture, pipelines, deployment topology | Architecture decisions change |
| `data-structures.mdc` | Data models, DTOs, schemas, serialization | New entities or schema changes |
| `file-locations.mdc` | Source code map, key files, module boundaries | New modules or significant restructuring |

**Key characteristic**: These are reference documentation, not enforcement rules. They use Apply Intelligently (`alwaysApply: false`) with clear descriptions so the agent loads them when working on relevant code.

---

## Naming Conventions

### File Names

- Use **kebab-case**: `api-conventions.mdc`, not `ApiConventions.mdc`
- Use **descriptive names**: `database-migration-workflow.mdc`, not `db.mdc`
- Use `.mdc` extension for files with frontmatter
- Use `.md` extension for simple markdown without frontmatter
- **One concept per file**: Do not combine unrelated concerns

### Folder Organization

- Keep the three top-level folders (`domain/`, `generic/`, `project/`) flat -- avoid deep nesting
- The only subfolder is `project/docs/` for the 4 documentation rules
- If a domain has many rules, you may create subfolders within `domain/` (e.g., `domain/react/`, `domain/database/`) but this is optional

---

## When to Use Each Folder: Quick Reference

| Question | Answer | Folder |
|---|---|---|
| Is it about a specific framework or technology? | Yes | `domain/` |
| Would it apply to any project regardless of tech stack? | Yes | `generic/` |
| Is it specific to this project's business logic or team? | Yes | `project/` |
| Is it documentation to help the agent understand the project? | Yes | `project/docs/` |
| Is it about how the rules themselves are organized? | Yes | `generic/` (as a meta rule) |

---

## The cursor-conventions Meta Rule

Every project should include a `generic/cursor-conventions.mdc` meta rule (Always Apply) that documents how the rules folder is organized. This helps the agent understand where to find relevant rules and when to consult each folder.

Example content:

```markdown
---
description: How Cursor rules are organized in this project
alwaysApply: true
---

# Cursor Rules Organization

Rules are organized in `.cursor/rules/` with three folders:

- **domain/**: Technology-specific rules (framework patterns, database conventions)
- **generic/**: Universal rules (git, coding standards, security)
- **project/**: Project-specific rules and documentation

Documentation rules in `project/docs/` describe this project's:
- API integration patterns (api-patterns.mdc)
- System architecture (architecture.mdc)
- Data models and schemas (data-structures.mdc)
- File locations and module map (file-locations.mdc)

Consult the appropriate documentation rule when you need project context.
```
