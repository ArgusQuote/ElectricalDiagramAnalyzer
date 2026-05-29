---
name: answering-vendor-questionnaires
description: Drafts and refines answers for vendor security questionnaires (SIG, CAIQ, custom enterprise security questionnaires) using Argus's published policy artifacts as primary citations. Use when a customer or prospect sends Argus a security or vendor risk questionnaire to fill out, when answers need to be softened, when stack-leak language needs to be scrubbed from a draft, or when mapping a question to the right Argus policy document. Operates on .docx files via python-docx.
---

# Answering Vendor Security Questionnaires

Drafts and refines answers for vendor / security / privacy questionnaires that
enterprise customers send to Argus, in Argus's house style: open about
infrastructure, conservative about contractual commitments, and silent about
internal software architecture.

## When to use this skill

- A customer or prospect has sent Argus a security questionnaire (SIG, CAIQ,
  vendor risk, privacy assessment, CCPA/CPRA assessment, etc.).
- An existing draft questionnaire needs softening, stack-leak scrubbing, or
  realignment with the Argus answering stance.
- A specific question needs to be mapped to one or more of the published
  Argus policy PDFs as the citation source.

## Inputs

- **The customer questionnaire**, typically a `.docx` file with a 2-row
  question-table layout. Argus's first such questionnaire was filed at
  `~/Documents/ArgusDocumentation/Vendor_Security_Questionnaire_FILLED_052526.docx`
  (the canonical reference for tone and depth).
- **The published Argus policy artifacts** at `~/Documents/ArgusDocumentation/`:
  - `Argus_Privacy_Policy.pdf`
  - `Argus_Terms_of_Service.pdf`
  - `Argus_IT_Security_Overview.pdf`
  - `Argus_Access_Control_Policy.pdf`
  - `Argus_Data_Retention_Deletion_Policy.pdf`
  - `Argus_Incident_Response_Policy.pdf`
  - `Argus_Vulnerability_Security_Management_Policy.pdf`
- **Project obscuring scope** -- this skill complements, but is narrower than,
  general project obscuring rules. The customer-facing answers in this skill
  are intentionally more open about cloud vendors than internal-doc obscuring
  rules suggest.
- The Python virtual environment per `.cursor/rules/domain/python-environment.mdc`,
  with `python-docx` installed.

## Steps

Copy this checklist and track progress:

```
Task Progress:
- [ ] Step 1: Inspect the questionnaire structure
- [ ] Step 2: Triage each question into one of four buckets
- [ ] Step 3: Draft answers using the Argus answering stance
- [ ] Step 4: Audit the draft for stack leaks and over-commitment
- [ ] Step 5: Apply edits via python-docx with pre/post verification
- [ ] Step 6: Hand back a list of placeholders Marco must fill personally
```

### Step 1: Inspect the questionnaire structure

Most enterprise questionnaires use a 2-row table per question, with the
response cell at row 1, column 0. Inside that cell is typically a 2-paragraph
structure: a bold "Response:" label paragraph, then a blank or filled answer
paragraph. Confirm the structure before writing any edits, because it varies
between vendors.

```python
from docx import Document
doc = Document("/path/to/Questionnaire.docx")
for idx, tbl in enumerate(doc.tables):
    nrows = len(tbl.rows)
    if nrows == 2:
        r0 = "\n".join(p.text for p in tbl.rows[0].cells[0].paragraphs)
        r1 = "\n".join(p.text for p in tbl.rows[1].cells[0].paragraphs)
        # r0 holds the question text; r1 starts with "Response:" if it's a
        # fillable answer cell.
```

Section banners are typically separate 1-row 1-column tables ("Section N. ...")
that interleave between question tables, so use them to map each question
table to a stable QID like `4.3` or `FT.7`.

### Step 2: Triage each question into one of four buckets

Walk every question and place it in exactly one bucket. The bucket determines
the drafting style.

| Bucket | When | What to do |
|---|---|---|
| **A. Cite a published policy** | The question maps cleanly to an Argus policy PDF (privacy, retention, incidents, access, vulnerability, ToS) | Quote or paraphrase the relevant policy section and direct the customer to the PDF in the review packet |
| **B. State current posture honestly + roadmap hedge** | Argus does not have the requested control today (no SOC 2, no DPA, no SAML, no SLA, no public status page, etc.) | Use the standard "Argus is an early-stage SaaS provider ... Argus may evaluate [X] as customer requirements mature" pattern |
| **C. Customer-specific decision** | The question requires Marco to choose a number, date, or commitment (e.g. specific liability cap, specific notification window) | Draft a placeholder like `[MARCO TO CONFIRM]` and surface it in the final hand-back list |
| **D. Identity / signature fields** | Top-of-doc identity rows, signature page, date completed | Leave blank for Marco to sign personally |

If you can't decide between A and B, prefer A and append the B hedge as a
secondary paragraph (e.g. "See Argus Privacy Policy, Section 7. Argus may
evaluate additional disclosures as customer requirements mature.").

### Step 3: Draft answers using the Argus answering stance

The Argus answering stance has four pillars. Apply all four to every answer.

#### Pillar 1: What is OK to disclose openly

- **Cloud and SaaS vendors by name and region**:
  - Anvil (web app, login, workflow management) -- region: `eu-west-2`
  - AWS (cloud infrastructure, document storage, processing) -- region: `us-east-2`
  - Paperspace (machine learning / GPU processing during infrastructure
    transition) -- region: `NY2`
  - Stripe (subscription billing, payment processing)
- **Legal entity**: "Argus Industrial Solutions, LLC", Texas, file number
  0805906427
- **Founders by name**: Logan Weber and Marco Soto
- **Universal contact email**: `argusquote@gmail.com` (one address for all
  security, privacy, support, and deletion requests today; do NOT invent
  separate `security@`, `privacy@`, or `support@` aliases)
- **Public website**: `https://argusquotes.com` (Privacy Policy and Terms of
  Service are linked from the website footer)
- **High-level workflow descriptors**: "in-house / proprietary extraction
  pipeline", "drawing-to-BOM workflow", "Generate -> Review -> Finalize",
  "panel schedule processing", "qualified-personnel review step"
- **Standards Argus does NOT follow**, named directly: "Argus does not sell
  customer data", "Argus does not currently submit uploaded drawings to
  third-party hosted AI services such as OpenAI, Anthropic, Google, or similar
  providers"

#### Pillar 2: What MUST stay obscured

Never name in any customer-facing answer:

- **ML / OCR libraries or model architectures**: PyTorch, TensorFlow, EasyOCR,
  OpenCV, Pillow, pikepdf, pypdfium2, Microsoft Table Transformer / TATR,
  MobileNetV2, detectron2, layoutparser, transformers (HF), torchvision, timm
- **Model family hints**: "transformer-based object-detection architecture",
  "ResNet backbone", specific DETR-family terminology
- **Internal source-control or repo-management specifics**: "deploy key" (use
  "host-restricted, read-only access credential" instead), GitHub or any
  source-control vendor by name, branch names, commit hashes
- **Internal file, function, or path names**: `uplink_server.py`, `WorkerSetup`,
  `vm_get_job_status`, `_cleanup_job_dir`, `~/jobs/`, `/home/paperspace/`,
  `/home/ubuntu/`, `RulesEngine5`, `BreakerTableParserAPIv11`, module names
  like `MLTableDetection` or `AnvilUplinkCode`, systemd unit names like
  `anvil-uplink.service` or `argus-uplink.service`
- **Specific OS / runtime versions**: Ubuntu 24.04, Python 3.10, CUDA 12.1,
  PyTorch 2.4.1, transformers 4.55.4 (use generic "modern Linux" /
  "supported Python runtime" if asked)
- **GPU SKU / hardware specifics**: A10G, A5000, RTX 500 Ada, vCPU counts,
  VRAM amounts
- **Defensive cloud-product namedrops**: avoid "no S3 / GCS / Azure Blob"
  even when answering in the negative. Use generic "no separate
  cloud-object-storage tier" instead. (The customer can deduce nothing useful
  from "we don't use service X" but it does signal stack familiarity.)

#### Pillar 3: Conservative hedging vocabulary

Use these phrases verbatim. They are the Argus tone:

- "Argus is an early-stage SaaS provider..."
- "...on a commercially reasonable basis."
- "...as customer requirements mature."
- "Argus may evaluate [X] as the platform and customer requirements mature."
- "...where commercially appropriate."
- "...subject to legal, security, billing, dispute-resolution, backup, and
  operational needs."
- "Argus does not currently maintain..."
- "Argus does not currently represent that..."
- "...no committed availability date is currently available."
- "...may be reviewed separately in writing."
- "...customer-specific content is not intentionally exposed to unrelated
  customers."
- "Customer-provided documents remain customer-owned."
- "Argus may use uploaded customer drawings and related processing data
  internally to operate, support, troubleshoot, secure, evaluate, and improve
  the Service unless otherwise agreed in writing with the customer."

#### Pillar 4: What NEVER goes in an answer (no matter how confident a draft is)

These are the lessons from the corrections applied to the first questionnaire.
A previous draft committed to all of these and every one was stripped during
review:

- **Specific numerical SLA percentages**: no "99.5% uptime", no "99.9%
  availability". Use "commercially reasonable basis".
- **Specific breach-notification windows**: no "within 72 hours of
  confirmation". Use "without undue delay after confirming relevant facts".
- **Specific patch SLAs**: no "7 days for critical CVEs / 30 days for high
  CVEs". Use "addressed on a risk-based basis, considering severity,
  exploitability, customer impact, and operational urgency".
- **Specific support response times**: no "4 business hours for Severity 1".
  Use "Argus prioritizes support requests based on severity, customer impact,
  service availability, data/security concerns, and operational urgency".
- **Specific breaking-change notice periods**: no "30 days minimum advance
  notice". Use "where commercially reasonable, Argus will seek to provide
  advance notice of material planned changes".
- **Specific RPO / RTO targets**: no "RPO <= 24 hours, RTO <= 4 hours". Use
  "Argus does not currently maintain formally published RPO/RTO commitments"
  + the early-stage hedge.
- **Specific liability caps in dollar amounts or month multiples**: defer to
  "subject to legal review during enterprise contract negotiation".
- **Specific target dates** for any roadmap item: never invent a date.
  "...no committed availability date is currently available" is the only
  acceptable form. Use a `[MARCO TO CONFIRM]` placeholder if the customer
  insists on one.
- **Specific cyber liability insurance carrier, limits, or policy details**.

If a draft from a prior agent contains any of the above, treat it as a
correction target.

### Step 4: Audit the draft for stack leaks and over-commitment

Before saving any edits, run two audits:

1. **Stack-leak audit**: scan every answer's text for the patterns in Pillar 2
   above. Use a regex sweep with case-insensitive word-boundary matching for
   each library/file/function name. Flag any hit.
2. **Over-commitment audit**: scan for the patterns in Pillar 4 above.
   Specifically look for any phrase of the form "within N [hours|days|business
   days|months]", any uptime percentage like "99.X%", and any specific dollar
   amount.

Cross-reference noise that is NOT a leak and should be ignored:

- Protocol versions (`TLS 1.2 / 1.3`, `SAML 2.0`, `SCIM 2.0`, `OIDC`,
  `Apache 2.0` / `MIT` / `BSD-3` license names) -- fine.
- Intra-document cross-references like "see 4.3" or "see Section 7" -- fine.
- Customer's own IdP product name in an SSO question (e.g. "Microsoft Entra
  ID will be supported once SAML / OIDC is generally available") -- fine,
  that's the customer's stack, not Argus's.
- Anvil and Stripe by name -- always allowed.

### Step 5: Apply edits via python-docx with pre/post verification

The response cell layout in the questionnaire docx is:

- Table row 1, column 0 (`tbl.rows[1].cells[0]`)
- Two paragraphs:
  - `paragraphs[0]` -- bold "Response:" label, single run, leave untouched
  - `paragraphs[1]` -- the answer, typically a single non-bold run

Substring replacements within `paragraphs[1].runs[0].text` preserve all
formatting. Whole-paragraph rewrites should set `runs[0].text = new_text` and
leave the run's formatting attributes alone.

Always:

1. **Take a backup** of the source docx before the first edit of a session
   (copy it to a sibling `.pre_edit.docx` filename).
2. **Pre-flight every edit**: confirm the `old_string` substring is present
   exactly once in the target paragraph before mutating. Skip silently if the
   `new_string` is already present (idempotency).
3. **Save and reload** the docx, then re-read the target cell to confirm the
   `new_string` is present and the `old_string` is gone. Fail loud on
   mismatch.

A reusable harness for this lives at `references/apply_edits_template.py`
(under this skill's folder).

### Step 6: Hand back a list of placeholders Marco must fill personally

Always finish a session by enumerating any `[MARCO TO CONFIRM]`, `[ARGUS LEGAL
ENTITY NAME]`, `[STATE / COUNTRY OF INCORPORATION]`, `[REGISTRATION NUMBER]`,
`[REGISTERED ADDRESS]`, `[FOUNDER NAME, TITLE]`, `[PRIMARY CONTACT NAME,
EMAIL]`, `[LLC / S-CORP / OTHER -- CONFIRM]`, `[PUBLISHED PRIVACY POLICY URL]`,
`[PUBLISHED TOS / MSA URL]`, or `[COOKIE / TRACKING DISCLOSURE URL]`
placeholders that remain in the document, along with the QID where each
appears. Marco fills these personally.

Identity rows at the top of the document (Vendor name, Primary contact, Date
completed, Version) and the signature page at the bottom are also Marco's
to fill -- never auto-populate them.

## Source citation map (question family -> policy doc)

When triaging a question into Bucket A, use this mapping to pick the citation
source:

| Question family | Cite |
|---|---|
| Privacy / personal information / consumer rights / cookies | `Argus_Privacy_Policy.pdf` |
| Acceptable use / IP ownership / customer responsibilities / billing | `Argus_Terms_of_Service.pdf` |
| Overall security posture / encryption / hosting | `Argus_IT_Security_Overview.pdf` |
| Authentication / authorization / role separation / production access | `Argus_Access_Control_Policy.pdf` |
| Data retention / deletion / customer data lifecycle | `Argus_Data_Retention_Deletion_Policy.pdf` |
| Incident response / breach notification / forensic preservation | `Argus_Incident_Response_Policy.pdf` |
| Vulnerability management / patch SLA / disclosure / pen-testing | `Argus_Vulnerability_Security_Management_Policy.pdf` |
| Subprocessor list / cloud regions / data residency | None of the policies covers this directly today. Use Pillar 1 disclosures inline + reference "the security/vendor review materials". |

If a question doesn't map cleanly to any of the above (e.g. SOC 2,
penetration testing, cyber liability insurance), it goes to Bucket B, not
Bucket A. Don't reach for a policy PDF that doesn't actually answer the
question.

## Reusable answer templates

Common question types come up in nearly every enterprise questionnaire. Use
these as a starting point and customize the wording to the specific question.

### Template: SOC 2 / ISO 27001 attestation

> None. Argus does not currently hold a SOC 2, ISO 27001, or similar
> third-party security attestation. Argus may evaluate formal
> security/compliance programs as customer requirements mature, but no
> committed date is currently available.

### Template: Signed DPA / standalone DPA

> Argus does not currently offer a standalone Data Processing Addendum for
> signature. Argus is an early-stage SaaS provider and currently addresses
> customer data handling, uploaded document ownership, internal
> service-improvement use, confidentiality, retention/deletion,
> subprocessors/infrastructure providers, and security posture through its
> Terms of Service, Privacy Policy, IT/Security Overview, and lightweight
> security/vendor policies. ... If a signed DPA is required for broader
> rollout or enterprise procurement, Argus can review the requirement with
> the customer.

### Template: SAML / OIDC SSO not currently supported

> Argus does not currently support customer-configurable SAML 2.0 or OIDC
> SSO. Authentication is currently provided through the Anvil-hosted
> application/login layer using individual user accounts. Argus may evaluate
> SAML 2.0, OIDC, or enterprise identity-provider integration as customer
> requirements mature, but no committed availability date is currently
> available.

### Template: Self-service tenant-wide deletion

> Self-service permanent purge is not currently available in the product.
> Customers may request deletion of uploaded drawings, generated BOMs,
> review artifacts, and related job data by contacting Argus at
> argusquote@gmail.com. Upon verified request, Argus will use commercially
> reasonable efforts to delete applicable customer-provided documents and
> related job data from active storage, subject to legal, security, billing,
> backup, dispute-resolution, and operational needs. Some residual copies
> may remain temporarily in backups, logs, archives, or system records
> according to normal technical retention practices until those records
> are overwritten, expired, or otherwise removed.

### Template: Breach notification timeline

> Argus does not currently maintain a standalone contractual
> breach-notification timeline or signed DPA breach-notification provision.
> If Argus determines that a security incident has materially affected a
> customer's data or account, Argus will use commercially reasonable
> efforts to notify the affected customer without undue delay after
> confirming relevant facts. Notification timing may depend on the nature
> of the incident, investigation status, provider involvement, legal
> requirements, and the need to prevent further harm.

### Template: Data residency (US-only or California-only)

> Argus does not currently offer customer-selectable data residency.
> Current hosting regions are: Anvil application/front-end and
> authentication: eu-west-2; AWS processing/storage infrastructure:
> us-east-2; Paperspace processing environment: NY2; Stripe billing/payment
> infrastructure: Stripe-managed. Uploaded drawings, generated processing
> artifacts, structured outputs, and related job files are primarily
> processed and stored in U.S.-hosted AWS and/or Paperspace environments.
> Because Argus currently uses Anvil-hosted application and authentication
> infrastructure in eu-west-2, Argus does not currently represent full
> U.S.-only residency across all application, account, authentication, and
> operational data.

### Template: Sale or sharing of personal information (CCPA/CPRA)

> Argus does not sell customer data or publicly disclose customer content.
> Argus does not currently use third-party advertising cookies, marketing
> pixels, cross-context behavioral advertising, or third-party ad targeting.
> Argus does not currently submit uploaded drawings or generated BOM
> outputs to third-party hosted AI/model providers such as OpenAI,
> Anthropic, Google, or similar providers for training, inference,
> retention, or model improvement. Argus is not aware of any current data
> flows that constitute a "sale" or "sharing" of personal information
> under CCPA/CPRA based on its current data practices.

## Stop conditions

Stop and ask Marco if any of the following come up:

- The questionnaire asks for a specific dollar amount of insurance coverage,
  a specific liability cap, a specific contractual notification window, or
  any other commitment that the templates above explicitly say to defer.
- A question demands a level of detail about Argus's internal architecture
  that cannot be answered honestly without naming a library, file, or
  function obscured by Pillar 2.
- A customer-specific exception is being negotiated (e.g. the customer
  requires a 24-hour breach notification clause in their MSA) -- contractual
  exceptions go through Marco, not the questionnaire response.
- The customer asks for a SOC 2 / pen-test report / continuous-compliance
  dashboard / status page URL -- these don't exist today, and the answer is
  Bucket B (state honestly + roadmap hedge), not "we will provide it".

## Recovery

- If a previous agent's draft has invented dates, percentages, or commitments
  that violate Pillar 4, treat it as a correction job. Strip the offending
  numbers and replace with the standard hedging vocabulary from Pillar 3.
  Do not preserve the previous draft's commitments out of respect for prior
  work -- the previous draft was not reviewed against this skill.
- If a stack-leak audit fires inside an answer that was supposedly already
  scrubbed, re-read the source code for the corresponding feature only as far
  as necessary to write a generic equivalent. Never paste internal source
  references into the questionnaire to fix a leak.

## References

- The corrected reference questionnaire (canonical example of Argus tone and
  depth): `~/Documents/ArgusDocumentation/Vendor_Security_Questionnaire_FILLED_052526.docx`
- The seven published Argus policy PDFs at `~/Documents/ArgusDocumentation/`.
- Project-level obscuring rules and committed-state conventions:
  `.cursor/rules/project/docs/known-issues.mdc` and
  `.cursor/rules/project/doc-conventions.mdc`. The customer-facing answers
  produced under this skill are intentionally more open about cloud vendors
  than internal-doc obscuring rules suggest, but track those documents for
  any narrowing that becomes necessary later.
