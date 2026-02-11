# Skill Authoring Best Practices

Sourced from: [Anthropic Platform Docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices), [agentskills.io Specification](https://agentskills.io/specification), [Builder.io Guide](https://builder.io/blog/agent-skills-rules-commands)

---

## SKILL.md Format (agentskills.io Specification)

Every skill is a directory containing a `SKILL.md` file with YAML frontmatter followed by Markdown content.

### Required Frontmatter

```yaml
---
name: skill-name
description: A description of what this skill does and when to use it.
---
```

### Optional Frontmatter Fields

| Field | Constraints | Purpose |
|---|---|---|
| `license` | Free text | License name or reference to bundled license file |
| `compatibility` | Max 500 chars | Environment requirements (system packages, network, etc.) |
| `metadata` | Key-value map | Arbitrary metadata (author, version, etc.) |
| `allowed-tools` | Space-delimited list | Pre-approved tools the skill may use (experimental) |
| `disable-model-invocation` | Boolean | When true, only included via explicit `/skill-name` invocation |

---

## Name Constraints

From agentskills.io specification:

- **Length**: 1-64 characters
- **Allowed characters**: Lowercase letters, numbers, and hyphens only (`a-z`, `0-9`, `-`)
- **No leading/trailing hyphens**: Must not start or end with `-`
- **No consecutive hyphens**: `--` is not allowed
- **Must match directory name**: The `name` field must equal the parent folder name

### Naming Convention (Anthropic Best Practices)

Prefer **gerund form** (verb + -ing) for clarity:

**Good**: `processing-pdfs`, `analyzing-spreadsheets`, `managing-databases`, `deploying-services`

**Acceptable**: Noun phrases (`pdf-processing`), action-oriented (`process-pdfs`)

**Avoid**: Vague names (`helper`, `utils`, `tools`), overly generic (`documents`, `data`, `files`)

---

## Description Requirements

From agentskills.io specification + Anthropic best practices:

- **Length**: 1-1024 characters, non-empty
- **Write in third person** (injected into system prompt):
  - Good: "Processes Excel files and generates reports"
  - Bad: "I can help you process Excel files"
  - Bad: "You can use this to process Excel files"
- **Include both WHAT and WHEN**:
  - WHAT: What the skill does (specific capabilities)
  - WHEN: When the agent should use it (trigger scenarios, keywords)
- **Include routing keywords**: Specific terms the agent matches against user tasks

### Good Description Examples

```yaml
# PDF Processing
description: Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction.

# Git Commit Helper
description: Generate descriptive commit messages by analyzing git diffs. Use when the user asks for help writing commit messages or reviewing staged changes.

# Code Review
description: Review code for quality, security, and maintainability following team standards. Use when reviewing pull requests, examining code changes, or when the user asks for a code review.
```

### Bad Description Examples

```yaml
description: Helps with documents.           # Too vague
description: Processes data.                  # No trigger keywords
description: Does stuff with files.           # Useless for routing
```

---

## The Concise-is-Key Principle (Anthropic Best Practices)

The context window is shared with conversation history, other skills, and the user's request. Every token competes for space.

**Default assumption**: The agent is already very smart. Only add context it does not already have.

Challenge each piece of information:
- "Does the agent really need this explanation?"
- "Can I assume the agent knows this?"
- "Does this paragraph justify its token cost?"

### Good (Concise) -- ~50 tokens

```markdown
## Extract PDF text

Use pdfplumber for text extraction:

\```python
import pdfplumber

with pdfplumber.open("file.pdf") as pdf:
    text = pdf.pages[0].extract_text()
\```
```

### Bad (Verbose) -- ~150 tokens

```markdown
## Extract PDF text

PDF (Portable Document Format) files are a common file format that contains
text, images, and other content. To extract text from a PDF, you'll need to
use a library. There are many libraries available for PDF processing, but we
recommend pdfplumber because it's easy to use and handles most cases well...
```

The concise version assumes the agent knows what PDFs are and how libraries work.

---

## Progressive Disclosure (Anthropic + agentskills.io)

Skills load information in three stages:

| Level | When Loaded | Token Cost | Content |
|---|---|---|
| **Metadata** | At startup (always) | ~100 tokens per skill | `name` and `description` from frontmatter |
| **Instructions** | When skill is activated | <5000 tokens recommended | Full SKILL.md body |
| **Resources** | As needed during execution | Effectively unlimited | Files in references/, scripts/, assets/ |

### Practical Guidelines

- Keep SKILL.md body **under 500 lines**
- Move detailed reference material to separate files in `references/`
- Move executable code to `scripts/`
- Keep file references **one level deep** from SKILL.md -- no nested chains
- Structure longer reference files (>100 lines) with a table of contents at the top

### Bad: Deeply Nested References

```
SKILL.md --> references/advanced.md --> references/details.md --> actual info
```

Agents may partially read nested files, resulting in incomplete information.

### Good: One Level Deep

```
SKILL.md
├── See [advanced.md](references/advanced.md)
├── See [reference.md](references/reference.md)
└── See [examples.md](references/examples.md)
```

---

## Degrees of Freedom (Anthropic Best Practices)

Match specificity to the task's fragility and variability:

### High Freedom (Text Instructions)

Use when multiple approaches are valid and decisions depend on context.

```markdown
## Code review process
1. Analyze the code structure and organization
2. Check for potential bugs or edge cases
3. Suggest improvements for readability
4. Verify adherence to project conventions
```

### Medium Freedom (Pseudocode/Templates)

Use when a preferred pattern exists but some variation is acceptable.

```markdown
## Generate report
Use this template and customize as needed:

\```python
def generate_report(data, format="markdown", include_charts=True):
    # Process data
    # Generate output in specified format
    # Optionally include visualizations
\```
```

### Low Freedom (Specific Scripts)

Use when operations are fragile, consistency is critical, or a specific sequence must be followed.

```markdown
## Database migration
Run exactly this script:

\```bash
python scripts/migrate.py --verify --backup
\```

Do not modify the command or add additional flags.
```

---

## Skill Directory Structure

```
skill-name/
├── SKILL.md              # Required: main instructions
├── scripts/              # Optional: executable code agents can run
│   ├── deploy.sh
│   └── validate.py
├── references/           # Optional: documentation loaded on demand
│   ├── REFERENCE.md
│   └── domain-guide.md
└── assets/               # Optional: templates, images, data files
    └── config-template.json
```

### Storage Locations

| Location | Scope |
|---|---|
| `~/.cursor/skills/skill-name/` | Personal -- available across all your projects |
| `.cursor/skills/skill-name/` | Project -- shared with anyone using the repository |

---

## Workflow Pattern (Anthropic Best Practices)

Break complex operations into clear sequential steps with a checklist:

```markdown
## Deployment workflow

Copy this checklist and track progress:

Task Progress:
- [ ] Step 1: Run tests
- [ ] Step 2: Build artifacts
- [ ] Step 3: Validate build
- [ ] Step 4: Deploy to staging
- [ ] Step 5: Verify staging
- [ ] Step 6: Deploy to production

**Step 1: Run tests**
Run: `./scripts/test.sh`
If tests fail, fix issues before continuing.

**Step 2: Build artifacts**
Run: `./scripts/build.sh --production`
...
```

## Feedback Loop Pattern (Anthropic Best Practices)

For quality-critical tasks, implement validation loops:

```markdown
## Document editing process

1. Make your edits
2. **Validate immediately**: `python scripts/validate.py output/`
3. If validation fails:
   - Review the error message
   - Fix the issues
   - Run validation again
4. **Only proceed when validation passes**
```

---

## 6-Point Skill Checklist (Builder.io)

Every skill should answer these six questions:

1. **Trigger (Description)**: When exactly should the agent load this?
2. **Inputs**: What info does it need from you or the repo before starting?
3. **Steps**: What is the procedure?
4. **Checks**: How do you prove it worked?
5. **Stop conditions**: When should it pause and ask for a human?
6. **Recovery**: What happens if a check fails?

---

## Common Failure Modes (Builder.io)

| Failure | Symptom | Fix |
|---|---|---|
| **The Encyclopedia** | Skill reads like a wiki page | Split into smaller files, use progressive disclosure |
| **The Everything Bagel** | Skill applies to every single task | It should be a rule, not a skill |
| **The Secret Handshake** | Agent never loads the skill | Rewrite description to match how users talk about tasks |
| **The Fragile Skill** | Breaks when repo changes | Move specifics into referenced files, keep skill logic procedural |

---

## Anti-Patterns to Avoid

### Time-Sensitive Information

Bad: "If you're doing this before August 2025, use the old API."

Good: Use a "Current method" section plus a collapsed "Old patterns (deprecated)" section.

### Inconsistent Terminology

Pick one term and use it throughout:
- Always "API endpoint" (not mixing "URL", "route", "path")
- Always "field" (not mixing "box", "element", "control")

### Too Many Options

Bad: "You can use pypdf, or pdfplumber, or PyMuPDF, or..."

Good: "Use pdfplumber for text extraction. For scanned PDFs requiring OCR, use pdf2image with pytesseract instead."

### Windows-Style Paths

Always use forward slashes: `scripts/helper.py` not `scripts\helper.py`
