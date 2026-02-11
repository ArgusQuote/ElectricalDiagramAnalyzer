---
name: reviewing-security
description: Reviews code for security vulnerabilities including injection attacks, credential exposure, authentication flaws, and unsafe data handling. Use when reviewing security-sensitive code, changes to auth or payment logic, or when the user asks to check for security issues.
---

# Reviewing Security

Perform a focused security review to identify vulnerabilities before they reach production.

## Inputs

- Code diff or file(s) to review, with focus on:
  - Authentication and authorization code
  - Payment or financial transaction logic
  - User data handling (PII, credentials, sessions)
  - API-facing code (endpoints, input parsing)
  - Database queries and data access
- Project security practices (from `.cursor/rules/generic/security-practices.mdc`)

## Steps

Copy this checklist and track progress:

```
Task Progress:
- [ ] Step 1: Check for injection vulnerabilities
- [ ] Step 2: Check authentication and authorization
- [ ] Step 3: Check for credential exposure
- [ ] Step 4: Check data handling
- [ ] Step 5: Check dependencies
- [ ] Step 6: Report findings
```

### Step 1: Check for injection vulnerabilities

Look for code that constructs queries or commands from user input:
- **SQL injection**: String concatenation in database queries instead of parameterized queries
- **XSS**: User input rendered in HTML without proper escaping
- **Command injection**: User input passed to shell commands or system calls
- **Path traversal**: User input used in file paths without sanitization
- **Template injection**: User input evaluated in template engines

### Step 2: Check authentication and authorization

Review access control logic:
- Are all protected endpoints checked for authentication?
- Is authorization verified for each operation (not just authentication)?
- Are there privilege escalation paths (e.g., user can modify another user's data)?
- Are session tokens generated with sufficient randomness?
- Is token expiration and refresh handled correctly?
- Are failed authentication attempts rate-limited?

### Step 3: Check for credential exposure

Search for hardcoded or leaked secrets:
- Hardcoded API keys, passwords, tokens, or connection strings in source code
- Secrets in log output, error messages, or API responses
- Credentials committed in configuration files
- Sensitive data in comments or debug code that should be removed
- Secrets in test fixtures that mirror real credentials

### Step 4: Check data handling

Review how sensitive data is processed:
- Is PII (personally identifiable information) encrypted at rest and in transit?
- Is sensitive data excluded from logs?
- Are user passwords hashed with a strong algorithm (bcrypt, argon2) and never stored in plaintext?
- Is data sanitized before storage?
- Are error messages generic to users (no internal details leaked)?
- Is HTTPS enforced for sensitive communications?

### Step 5: Check dependencies

Review third-party code security:
- Are any dependencies known to have CVEs? (Check with project's dependency audit tool)
- Are dependency versions pinned to avoid supply chain attacks?
- Are new dependencies from well-maintained, reputable sources?

### Step 6: Report findings

For each security finding, provide:

1. **File and line**: Specific location of the vulnerability
2. **Severity**: Critical / High / Medium / Low
3. **Category**: Injection, Auth, Credentials, Data Handling, Dependencies
4. **Description**: What the vulnerability is and how it could be exploited
5. **Fix**: A concrete code change to remediate the issue

Organize findings from highest to lowest severity.

## Checks

- All Critical and High severity findings are addressed
- No hardcoded secrets remain in the codebase
- All user input paths have appropriate validation and sanitization

## Stop Conditions

- If a vulnerability requires domain-specific knowledge (e.g., which users should have access to which resources), ask the user to confirm the intended access model
- If a potential vulnerability is in a third-party dependency, recommend updating the dependency rather than patching around it
- If a finding may be a false positive, flag it with an explanation and let the user decide

## Recovery

- If a finding is confirmed as a false positive, document why it is safe (add a comment in code explaining the security consideration)
- If a Critical vulnerability is found in production code, recommend an immediate remediation plan and suggest notifying the security team
