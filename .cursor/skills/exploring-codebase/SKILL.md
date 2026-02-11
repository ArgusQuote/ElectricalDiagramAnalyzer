---
name: exploring-codebase
description: Explores and summarizes a codebase's structure, entry points, conventions, and run commands. Use when onboarding to a new project, when the user asks "explain this repo," "how does this work," or needs to understand project organization.
---

# Exploring Codebase

Systematically explore and document a project's structure so the agent and developer can work effectively within it.

## Inputs

- Access to the project's file tree and configuration files
- Any existing documentation (README, CLAUDE.md, `.cursor/rules/project/docs/`)

## Steps

Copy this checklist and track progress:

```
Task Progress:
- [ ] Step 1: Scan project structure
- [ ] Step 2: Read configuration files
- [ ] Step 3: Identify entry points and key modules
- [ ] Step 4: Discover run/build/test commands
- [ ] Step 5: Document observed conventions
- [ ] Step 6: Update project documentation rules
```

### Step 1: Scan project structure

List the top-level directory structure. Identify the project layout pattern:
- Monorepo vs single project
- Source code location (src/, lib/, app/, etc.)
- Test location (tests/, test/, __tests__/, etc.)
- Configuration location (config/, etc.)
- Documentation location (docs/, etc.)

### Step 2: Read configuration files

Read the project's key configuration files to understand:
- **Build system**: package.json, pyproject.toml, Cargo.toml, build.gradle, Makefile, etc.
- **Dependencies**: lockfiles, requirements files, dependency manifests
- **Environment**: .env.example, docker-compose.yml, Dockerfile
- **Linting/formatting**: .eslintrc, .prettierrc, ruff.toml, .editorconfig, etc.
- **CI/CD**: .github/workflows/, .gitlab-ci.yml, Jenkinsfile, etc.

### Step 3: Identify entry points and key modules

Find the main entry points:
- Application entry point (main, index, app, server, etc.)
- Key modules and their responsibilities
- Public API surface (routes, endpoints, exports)
- Shared utilities or common libraries

### Step 4: Discover run/build/test commands

Determine the commands for:
- **Development**: How to start the project locally
- **Build**: How to create a production build
- **Test**: How to run the test suite (full and single-file)
- **Lint**: How to check code style and quality
- **Type check**: How to run static type checking (if applicable)

### Step 5: Document observed conventions

Note patterns observed in the codebase:
- Naming conventions (files, functions, classes, variables)
- Code organization patterns (layered architecture, feature folders, etc.)
- Error handling patterns
- Testing patterns (unit, integration, e2e)
- Import/module resolution patterns

### Step 6: Update project documentation rules

If `.cursor/rules/project/docs/` exists, offer to update:
- `file-locations.mdc` with the directory structure and key files
- `architecture.mdc` with component relationships discovered
- `data-structures.mdc` with core entities found
- `api-patterns.mdc` with API integration patterns observed

## Checks

- Summary accurately reflects the actual project structure (verify by spot-checking key files)
- Documented run/build/test commands execute without error
- No guesses presented as facts -- only document what was observed

## Stop Conditions

- If the project has no clear entry point or build system, ask the user for guidance
- If the project requires credentials or external services to explore fully, ask the user how to proceed
- If the codebase is a monorepo with many subprojects, ask which subproject to focus on first

## Recovery

- If a run command fails, document the failure and ask the user whether the project needs setup steps first
- If configuration files reference tools not installed locally, note the missing tools and continue with what is available
