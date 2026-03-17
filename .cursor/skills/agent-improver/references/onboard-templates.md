# Project Onboarding Templates

Starter templates for the 4 documentation rule files created during Project Onboard mode, plus the 2 baseline generic rules.

---

## Documentation Rules (project/docs/)

### api-patterns.mdc

```markdown
---
description: Documentation of external API integrations, connection patterns, authentication flows, and error handling conventions. Use when working with API clients, external service calls, REST endpoints, or HTTP communication code.
alwaysApply: false
---

# API Patterns

## External Service Integrations

<!-- List each external service this project connects to -->

### [Service Name]
- **Base URL**: [URL or config property reference]
- **Authentication**: [API key / OAuth2 / JWT / mTLS]
- **Client class**: [path to the client implementation]
- **Error handling**: [retry policy, circuit breaker, fallback behavior]

## Connection Patterns

### REST Client Pattern
<!-- Describe the standard pattern for creating REST clients in this project -->

### Authentication Flow
<!-- Describe how authentication tokens are obtained, refreshed, and passed -->

### Error Handling Convention
<!-- Describe the standard error handling approach for external calls:
     - Retry policy (which errors, how many retries, backoff)
     - Circuit breaker configuration
     - Fallback behavior
     - Error logging and alerting -->

### Timeout Configuration
<!-- Standard timeout values and where they are configured -->

## API Versioning
<!-- How API versions are managed in this project -->
```

### architecture.mdc

```markdown
---
description: System architecture documentation including pipeline flows, component relationships, deployment topology, and key design decisions. Use when making architectural decisions, understanding system flow, or modifying pipeline components.
alwaysApply: false
---

# System Architecture

## Overview
<!-- One paragraph describing what this system does and its primary purpose -->

## Component Diagram
<!-- List the major components and their relationships.
     Use a text diagram or describe the flow:
     
     [Component A] --> [Component B] --> [Component C]
                                    \--> [Component D]
-->

## Key Components

### [Component Name]
- **Purpose**: [what it does]
- **Location**: [path in codebase]
- **Dependencies**: [what it depends on]
- **Consumers**: [what depends on it]

## Data Pipelines

### [Pipeline Name]
- **Trigger**: [what initiates this pipeline]
- **Steps**: [ordered list of processing steps]
- **Output**: [what the pipeline produces]
- **Error handling**: [what happens when a step fails]

## Deployment Topology
<!-- Describe how the system is deployed:
     - Environments (dev, staging, production)
     - Infrastructure (containers, serverless, VMs)
     - Key configuration differences between environments -->

## Key Design Decisions

### [Decision Title]
- **Decision**: [what was decided]
- **Context**: [why this decision was made]
- **Alternatives considered**: [what else was evaluated]
- **Consequences**: [trade-offs accepted]
```

### data-structures.mdc

```markdown
---
description: Documentation of core data models, DTOs, pipeline payloads, database schemas, and serialization formats. Use when working with data models, creating new entities, modifying schemas, or understanding data flow.
alwaysApply: false
---

# Data Structures

## Core Entities

### [Entity Name]
- **Location**: [path to the entity class]
- **Table**: [database table name]
- **Purpose**: [what this entity represents]
- **Key fields**:
  - `id` -- [type, generation strategy]
  - `[field]` -- [type, constraints, description]
- **Relationships**: [references to other entities]

## DTOs and Payloads

### [DTO Name]
- **Location**: [path to the DTO class]
- **Used by**: [which endpoint or pipeline uses this]
- **Fields**:
  - `[field]` -- [type, validation rules, description]

## Database Schema Conventions
<!-- Describe the naming and structural conventions:
     - Table naming (singular/plural, prefix, case)
     - Column naming (snake_case, camelCase)
     - Index conventions
     - Migration approach (Flyway, Liquibase, etc.) -->

## Serialization
<!-- Describe how data is serialized:
     - JSON conventions (property naming, null handling)
     - Date/time format
     - Custom serializers if any -->

## Data Flow
<!-- Describe how data flows through the system:
     Request DTO --> Service Layer --> Entity --> Database
     Database --> Entity --> Response DTO --> Client -->
```

### file-locations.mdc

```markdown
---
description: Map of source code file locations, directory structure, key files and their purposes, and module boundaries. Use when navigating the codebase, finding where functionality lives, or understanding project organization.
alwaysApply: false
---

# File Locations

## Directory Structure

```
src/
├── main/
│   ├── <module>/            # Source code (adapt path to your language)
│   │   ├── api/             # REST/API handlers
│   │   ├── service/         # Business logic
│   │   ├── model/           # Domain entities
│   │   ├── dto/             # Data transfer objects
│   │   ├── config/          # Configuration
│   │   └── client/          # External service clients
│   ├── resources/           # Static assets, config files
│   │   └── db/migration/    # Database migrations (if applicable)
│   └── docker/              # Dockerfiles
└── test/                    # Test classes mirror main structure
```

## Key Files

| File | Purpose |
|---|---|
| <!-- e.g., pyproject.toml, package.json --> | Build configuration and dependencies |
| <!-- e.g., .env, config.yaml --> | Application configuration |
| <!-- Add key files as they are created --> | |

## Module Boundaries

### [Module Name]
- **Path/namespace**: [e.g., src/module-name or package.path]
- **Purpose**: [what this module handles]
- **Entry point**: [main class or resource]
- **Dependencies**: [other modules it uses]

## Conventions
<!-- Describe where new files should be placed:
     - New API/endpoint --> api/
     - New business logic --> service/
     - New entity/model --> model/
     - New external client --> client/ -->
```

---

## Baseline Generic Rules

### generic/cursor-conventions.mdc

```markdown
---
description: How Cursor rules are organized in this project and when to consult each folder
alwaysApply: true
---

# Cursor Rules Organization

Rules in `.cursor/rules/` are organized into three categories:

## domain/
Technology-specific rules for the frameworks and tools used in this project. Consult when working with framework-specific code patterns.

## generic/
Universal rules that apply regardless of technology. Includes coding standards, git conventions, security practices, and this meta rule.

## project/
Project-specific rules and living documentation:
- **project/docs/api-patterns.mdc** -- External API integrations and connection patterns
- **project/docs/architecture.mdc** -- System architecture and pipeline flows
- **project/docs/data-structures.mdc** -- Data models, DTOs, and schema conventions
- **project/docs/file-locations.mdc** -- Source code map and module boundaries

Consult the appropriate docs/ rule when you need context about how this project works.
```

### generic/coding-standards.mdc

```markdown
---
description: Core coding standards that apply across all code in this project
alwaysApply: true
---

# Coding Standards

## Naming
- Use clear, descriptive names that reveal intent
- Avoid abbreviations unless universally understood (e.g., URL, HTTP, ID)
- Be consistent with naming patterns across the codebase

## Error Handling
- Never silently swallow exceptions
- Always log errors with sufficient context for debugging
- Use specific exception types, not generic Exception

## Code Quality
- No magic numbers or strings -- use named constants
- Keep methods focused on a single responsibility
- Prefer early returns to reduce nesting

## Documentation
- Document WHY, not WHAT (code shows what, comments explain why)
- Keep comments up to date when changing code
- Add a brief doc comment to public methods explaining purpose and parameters
```
