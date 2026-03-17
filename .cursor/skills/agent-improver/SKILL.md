---
name: agent-improver
description: Create and improve Cursor skills and rules using best practices from authoritative sources (Anthropic, agentskills.io, Cursor docs, Builder.io, Atlan). Use when the user wants to create a skill, create a rule, onboard a new project with rule scaffolding, or review and improve existing skills and rules.
---
# Agent Improver

A meta-skill for creating and improving Cursor skills and rules using best practices from 6 authoritative sources.

## Detect Mode

Determine which mode to use based on the user's request:

| User Intent | Mode |
|---|---|
| Create a new skill, write a SKILL.md | **Create Skill** |
| Create a new rule, write an .mdc file | **Create Rule** |
| Set up a new project, onboard, scaffold rules | **Project Onboard** |
| Review, improve, audit existing skills or rules | **Review and Improve** |

If the intent is ambiguous, ask the user which mode they need.

---

## Mode 1: Create Skill

Follow these steps to create a new skill that conforms to the Agent Skills open standard.

### Step 1: Gather Requirements

Use the AskQuestion tool to determine:
1. **Purpose**: What task or workflow should this skill help with?
2. **Location**: Personal (`~/.cursor/skills/`) or project (`.cursor/skills/`)?
3. **Trigger scenarios**: When should the agent apply this skill?
4. **Domain knowledge**: What specialized info does the agent need that it would not already know?

If previous conversation context provides answers, infer them instead of asking.

### Step 2: Research Domain

Use web search to find authoritative documentation for the skill's domain. Prefer official docs, specs, and well-known engineering blogs. Capture key patterns, constraints, and examples.

### Step 3: Design the Skill

Read [skills-best-practices.md](references/skills-best-practices.md) for the full specification, then apply these requirements:

**Name**: 1-64 chars, lowercase + hyphens only, no leading/trailing/consecutive hyphens. Must match directory name. Prefer gerund form (e.g., `processing-pdfs`, `deploying-services`).

**Description**: Write in third person. Include both WHAT the skill does and WHEN to use it. Include specific keywords for agent routing. Max 1024 chars.

**Body**: Apply the concise-is-key principle -- only include context the agent does not already have. Challenge every section: "Does this justify its token cost?"

**Degrees of freedom**:
- **High** (text instructions): Multiple valid approaches, context-dependent decisions
- **Medium** (pseudocode/templates): Preferred pattern with acceptable variation
- **Low** (specific scripts): Fragile operations, consistency critical

**Progressive disclosure**: Keep SKILL.md body under 500 lines. Move detailed reference material to `references/`. Move executable code to `scripts/`. Keep all file references one level deep from SKILL.md.

### Step 4: Create the Files

1. Create the skill directory: `<location>/<skill-name>/`
2. Write SKILL.md with frontmatter and body
3. Create `references/`, `scripts/`, `assets/` directories only if needed
4. Write any supporting files

### Step 5: Validate

Run through this checklist:
- [ ] Name matches directory, follows constraints
- [ ] Description is third person, includes WHAT + WHEN, has routing keywords
- [ ] SKILL.md body is under 500 lines
- [ ] All file references are one level deep
- [ ] No time-sensitive information
- [ ] Consistent terminology throughout
- [ ] **Trigger**: When should the agent load this?
- [ ] **Inputs**: What info does it need before starting?
- [ ] **Steps**: What is the procedure?
- [ ] **Checks**: How do you prove it worked?
- [ ] **Stop conditions**: When should it pause and ask a human?
- [ ] **Recovery**: What happens if a check fails?

---

## Mode 2: Create Rule

Follow these steps to create a new Cursor rule that conforms to official docs and industry best practices.

### Step 1: Determine Rule Type

Read [rules-best-practices.md](references/rules-best-practices.md) for the full guide, then apply the Builder.io litmus test:

> "Would you want this instruction to apply even when you are not thinking about it?"

**Yes** --> Always Apply rule (`alwaysApply: true`)
**No, but it is file-specific** --> Apply to Specific Files (set `globs`)
**No, it is situational** --> Apply Intelligently (`alwaysApply: false`, no globs)
**No, it is a one-off tool** --> Apply Manually (user invokes with @rule-name)

### Step 2: Classify with the Atlan Framework

Determine which category the rule belongs to:

| Category | Rule Type | Examples |
|---|---|---|
| **Project Rules** | Always | Project overview, architecture, structure, glossary |
| **Tech Stack Rules** | Always or Agent Requested | Language conventions, framework patterns, API standards |
| **Micro Workflow Rules** | Agent Requested or Manual | Logging, auth wrappers, feature flags, migrations |
| **Meta Rules** | Always | Rule prioritization, conflict resolution, conventions |

### Step 3: Determine Folder Placement

Read [folder-convention.md](references/folder-convention.md), then place the rule:

| Content Type | Folder |
|---|---|
| Framework/technology patterns (React, API layer, etc.) | `.cursor/rules/domain/` |
| Universal rules (git, security, docs, meta) | `.cursor/rules/generic/` |
| This-project-only rules and docs | `.cursor/rules/project/` |
| Project documentation for agent context | `.cursor/rules/project/docs/` |

### Step 4: Write the Rule

Create an `.mdc` file with YAML frontmatter:

```
---
description: [Clear description of what this rule does and when to apply it]
globs: [optional file pattern, e.g., **/*.ts, **/*.py]
alwaysApply: [true or false]
---

# Rule Title

[Rule content here]
```

**Writing principles**:
- Be assertive, not suggestive: "Always use X" not "Consider using X"
- One concept per rule
- Under 50-80 lines ideally, max 500
- Include concrete examples with good/bad patterns
- Reference files instead of copying content
- Place the most important constraints last (LLMs weight end of input)

### Step 5: Validate

- [ ] File is `.mdc` in `.cursor/rules/<subfolder>/`
- [ ] Frontmatter has correct `description`, `globs`, `alwaysApply`
- [ ] Description is clear enough for Apply Intelligently rules to trigger
- [ ] Content is under 500 lines, ideally under 80
- [ ] Uses assertive language
- [ ] One focused concept
- [ ] Includes concrete examples
- [ ] No copied style guides or command docs the agent already knows

---

## Mode 3: Project Onboard

Scaffold the `.cursor/rules/` folder structure and documentation rules for a new project.

### Step 1: Create Folder Structure

Create the following directories:
```
.cursor/rules/
├── domain/
├── generic/
└── project/
    └── docs/
```

### Step 2: Ask About Domain

Use AskQuestion to determine the project's primary technology domain (e.g., Node.js, Python, React). This determines what goes in the `domain/` folder later.

### Step 3: Create Documentation Rules

Read [onboard-templates.md](references/onboard-templates.md) for the full templates, then create these 4 files in `.cursor/rules/project/docs/`:

**api-patterns.mdc** -- External service connection patterns
- `alwaysApply: false`
- Description: "Documentation of external API integrations, connection patterns, authentication flows, and error handling conventions. Use when working with API clients, external service calls, REST endpoints, or HTTP communication code."

**architecture.mdc** -- System architecture and pipelines
- `alwaysApply: false`
- Description: "System architecture documentation including pipeline flows, component relationships, deployment topology, and key design decisions. Use when making architectural decisions, understanding system flow, or modifying pipeline components."

**data-structures.mdc** -- Pipeline data structure documentation
- `alwaysApply: false`
- Description: "Documentation of core data models, DTOs, pipeline payloads, database schemas, and serialization formats. Use when working with data models, creating new entities, modifying schemas, or understanding data flow."

**file-locations.mdc** -- Code file locations and descriptions
- `alwaysApply: false`
- Description: "Map of source code file locations, directory structure, key files and their purposes, and module boundaries. Use when navigating the codebase, finding where functionality lives, or understanding project organization."

### Step 4: Create Baseline Generic Rules

Create `generic/cursor-conventions.mdc` (Always Apply):
- Documents how the `.cursor/rules/` folder is organized
- Explains the domain/generic/project structure
- Tells the agent when to consult each folder

Create `generic/coding-standards.mdc` (Always Apply):
- Basic coding standards: clear naming, error handling, no magic numbers
- Keep minimal -- only add what the agent would not already know

### Step 5: Confirm and Next Steps

Tell the user:
1. The folder structure is ready
2. The 4 doc rule files have starter templates -- fill them in as the project develops
3. Domain-specific rules should be added to `domain/` as needed
4. Use agent-improver in Create Rule mode to add more rules

---

## Mode 4: Review and Improve

Analyze existing skills and rules for quality issues against authoritative best practices.

### Step 1: Scan

Scan the project for:
- Skills in `.cursor/skills/` and `~/.cursor/skills/`
- Rules in `.cursor/rules/`
- Any `.cursorrules` legacy files (recommend migration)

### Step 2: Check Skills

For each skill, check against [skills-best-practices.md](references/skills-best-practices.md):

| Check | Source | What to Look For |
|---|---|---|
| Name valid | agentskills.io spec | Lowercase, hyphens, 1-64 chars, matches directory |
| Description quality | Anthropic docs | Third person, WHAT + WHEN, routing keywords |
| Conciseness | Anthropic docs | Only includes what agent does not already know |
| Line count | agentskills.io spec | SKILL.md under 500 lines |
| Progressive disclosure | Anthropic docs | Heavy content in references/, not SKILL.md |
| Reference depth | Anthropic docs | All refs one level deep from SKILL.md |
| No time-sensitive info | Anthropic docs | No date-dependent instructions |
| Consistent terminology | Anthropic docs | One term per concept throughout |
| Not an "Everything Bagel" | Builder.io | If it applies to every task, it should be a rule |
| Not a "Secret Handshake" | Builder.io | Description must match how users talk about tasks |

### Step 3: Check Rules

For each rule, check against [rules-best-practices.md](references/rules-best-practices.md):

| Check | Source | What to Look For |
|---|---|---|
| Correct rule type | Builder.io litmus test | alwaysApply matches the rule's nature |
| Correct category | Atlan framework | Project/Tech Stack/Micro Workflow/Meta |
| Correct folder | Folder convention | domain/generic/project placement |
| Frontmatter valid | Cursor docs | description, globs, alwaysApply all correct |
| Line count | Cursor docs + Atlan | Under 500, ideally under 80 |
| Assertive language | Atlan | "Always X" not "Consider X" |
| Focused scope | Atlan | One concept per rule |
| Concrete examples | Cursor docs | Good/bad patterns shown |
| No style guide copies | Cursor docs | Reference, don't copy |
| Glob accuracy | Cursor docs | Patterns match intended files |

### Step 4: Report

For each issue found, provide:
1. **File**: Path to the skill or rule
2. **Issue**: What is wrong and which source identifies it
3. **Before**: Current problematic content
4. **After**: Concrete fix example
5. **Priority**: Critical (must fix) / Suggestion (should fix) / Nice-to-have

---

## Authoritative Sources

This skill's guidance is sourced from these verified references:

| Source | URL | Covers |
|---|---|---|
| Anthropic Skill Best Practices | platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices | Skill authoring, progressive disclosure, naming |
| Agent Skills Spec | agentskills.io/specification | SKILL.md format, frontmatter constraints |
| Cursor Rules Docs | cursor.com/docs/context/rules | Rule types, .mdc format, best practices |
| Cursor Skills Docs | cursor.com/docs/context/skills | Skill directories, discovery, invocation |
| Builder.io Guide | builder.io/blog/agent-skills-rules-commands | Skills vs rules vs commands decision framework |
| Atlan Framework | blog.atlan.com/engineering/cursor-rules/ | 4-category rule taxonomy, writing philosophy |

For detailed guidance, see:
- [Skill authoring best practices](references/skills-best-practices.md)
- [Rule authoring best practices](references/rules-best-practices.md)
- [How skills and rules connect](references/connecting-skills-and-rules.md)
- [Folder structure convention](references/folder-convention.md)
- [Project onboarding templates](references/onboard-templates.md)
