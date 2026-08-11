# 🧑‍💻 Development Guide

## 📋 Purpose and Scope

This guide establishes the development, testing, security, and deployment practices for the AI-Powered Alcohol Label Verification prototype. The supplied materials define outcomes and constraints but do not identify the repository's actual framework, package manager, or command names. Commands below use explicit placeholders where repository inspection is required; replace those placeholders with verified project commands rather than documenting guesses.

## ✅ Prerequisites

The development environment should provide:

* A supported runtime version declared by the repository.
* The repository's selected package manager and lock file.
* Git.
* Docker or an equivalent container runtime when local service emulation is supported.
* Access to an approved Azure subscription or a documented local OCR test double.
* Test label images containing ordinary, imperfect, and deliberately noncompliant examples.

Do not place Azure keys, storage credentials, real application data, or production label images in the repository.

## 🗂️ Recommended Repository Structure

Adapt names to the implementation while preserving separation of concerns.

```text
.
├── src/
│   ├── api/                 # HTTP routes and request/response models
│   ├── application/         # Verification and batch use cases
│   ├── domain/              # Field models, comparison rules, statuses
│   ├── infrastructure/      # OCR, temporary storage, telemetry adapters
│   └── ui/                  # Accessible user interface
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   ├── accessibility/
│   └── performance/
├── docs/
│   └── api/
├── samples/                 # Synthetic, non-sensitive test artifacts
├── .env.example             # Names only; no secrets
├── Dockerfile
└── README.md
```

## ⚙️ Local Configuration

Use environment variables or the platform's secure configuration provider. Keep configuration names stable across local, test, and Azure deployments.

| Setting | Purpose | Secret |
|---|---|---:|
| `APP_ENVIRONMENT` | Select development, test, or production behavior | No |
| `OCR_ENDPOINT` | Approved Azure or local container endpoint | No |
| `OCR_CREDENTIAL` | OCR access credential when managed identity is unavailable | Yes |
| `TEMP_STORAGE_URI` | Short-lived processing location | Usually no |
| `MAX_FILE_BYTES` | Per-file upload limit | No |
| `MAX_BATCH_ITEMS` | Maximum labels per batch | No |
| `TEMP_RETENTION_MINUTES` | Maximum temporary-content lifetime | No |
| `FUZZY_MATCH_THRESHOLD` | Default threshold for eligible text fields | No |
| `TELEMETRY_ENABLED` | Enable operational telemetry | No |
| `LOG_LEVEL` | Minimum log severity | No |

Provide `.env.example` with safe placeholder values. Ensure `.env`, local databases, temporary uploads, downloaded secrets, and test outputs containing label text are ignored by version control.

## 🏃 Standard Development Workflow

1. Create a feature branch from the current integration branch.
2. Install dependencies from the lock file.
3. Copy the example configuration and supply local, non-production values.
4. Run formatting, static analysis, and the complete fast test suite before changing code.
5. Implement the smallest complete change at the correct layer.
6. Add or update unit, integration, contract, accessibility, and performance tests as applicable.
7. Regenerate API documentation from the implementation when schemas change.
8. Run the full verification suite.
9. Review the diff for secrets, generated binaries, test artifacts, and unrelated changes.
10. Document assumptions, limitations, and operational impact in the pull request.

Repository documentation should expose concrete commands for these capabilities:

```text
<install-command>
<format-check-command>
<lint-command>
<type-check-command>
<unit-test-command>
<integration-test-command>
<run-command>
```

Do not replace these placeholders until the repository's actual tooling has been verified.

## 🧩 Core Domain Contracts

### Field result

Every field comparison should produce:

* Field name.
* Expected application value.
* Observed label value.
* Normalized comparison values where applicable.
* Status.
* Severity.
* Extraction confidence.
* Comparison method.
* Human-readable explanation.
* Evidence location.

Original values must remain available even when comparison values are normalized.

### Status rules

Use only `pass`, `fail`, `manual_review`, `not_found`, and `not_applicable`. Low confidence must not produce an automatic pass. A missing required field must not be represented as an empty successful value.

### Comparison strategies

Implement independent, testable strategies rather than a single generic string comparator:

* Text-equivalence strategy for case, whitespace, punctuation, and typographic variants.
* Controlled-vocabulary strategy for beverage classes and types.
* Numeric-unit strategy for ABV, proof, and net contents.
* Structured-name/address strategy for producer and bottler data.
* Canonical-country strategy for origin fields.
* Exact-text-and-format strategy for the government warning.

Thresholds must be configuration values with documented rationale. Store the method and threshold used in diagnostic output so results are reproducible.

## 🔤 OCR Adapter Design

The domain and application layers must depend on an OCR interface, not directly on an Azure SDK. The adapter should translate provider output into an internal model containing:

* Page dimensions and units.
* Full text.
* Lines, words, and reading order.
* Bounding regions.
* Confidence values.
* Available font-style evidence.
* Provider model/version metadata for diagnostics.

This boundary permits deterministic test doubles and prevents provider response types from spreading through the rules engine. Azure Document Intelligence produces structured JSON for document text and related elements; consult the current [Microsoft documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/faq?view=doc-intel-4.0.0) when selecting a model and SDK version.

## 🖼️ Image Preprocessing

Preprocessing should be a measurable pipeline with each transformation independently enabled and tested:

1. Decode and verify the image.
2. Apply orientation metadata safely.
3. Detect gross rotation.
4. Generate an OCR working copy at an appropriate resolution.
5. Apply conservative contrast correction.
6. Apply perspective correction only when boundary confidence is sufficient.
7. Preserve the unmodified original in short-lived working memory for evidence display.

Never overwrite the original evidence or apply transformations that can remove punctuation, change characters, or create artificial text strokes without retaining the original result path.

## ⚖️ Government Warning Rule

Treat the warning validator as a high-assurance rules component. TTB guidance states that `GOVERNMENT WARNING` must be uppercase and bold, the remainder must form a continuous statement, and the warning must be separate from other information. See [TTB's current health-warning guidance](https://www.ttb.gov/regulated-commodities/beverage-alcohol/distilled-spirits/ds-labeling-home/ds-health-warning).

Required tests include:

* Exact valid text.
* Missing heading.
* Heading in title case.
* Heading not bold when style information is dependable.
* Changed, missing, duplicated, or reordered words.
* Punctuation differences.
* Text interrupted by unrelated label content.
* Low-confidence OCR.
* Cropped warning.
* Inconclusive type size, contrast, or formatting evidence.

The last four conditions should generally require manual review instead of asserting compliance.

## 🧪 Test Strategy

### Unit tests

Unit tests should cover normalization, every comparison policy, severity mapping, status aggregation, input validation, archive safety, CSV escaping, and temporary-file cleanup. Tests must include Unicode punctuation and mixed-case text.

### Integration tests

Integration tests should exercise the OCR adapter, API serialization, temporary storage, cancellation, dependency failures, and cleanup. Use synthetic or approved non-sensitive fixtures.

### Contract tests

Validate requests and responses against the published OpenAPI schema. Confirm enum values, optionality, content types, error documents, and backwards-compatible additive changes.

### Accessibility tests

Combine automated WCAG checks with keyboard-only review, screen-reader smoke testing, 200-percent zoom, high-contrast mode, and verification that status is not communicated through color alone.

### Performance tests

Use a documented corpus split by file size, resolution, skew, glare, beverage type, and number of fields. Capture:

* Preprocessing time.
* OCR time.
* Extraction and comparison time.
* End-to-end server time.
* Browser-visible response time.
* Batch throughput and maximum concurrency.
* P50, P95, and P99 latency.

The stakeholder acceptance target is under five seconds per ordinary label. Report hardware, Azure region, service tier, concurrency, warm/cold state, and corpus characteristics with every benchmark.

## 📚 Test Corpus Governance

Maintain a versioned manifest for every test image with:

* Synthetic or approved source classification.
* Beverage type.
* Expected fields.
* Expected rule outcomes.
* Image-quality conditions.
* Licensing or creation provenance.
* Whether the artifact may be committed to the repository.

Include ordinary compliant labels, minor typographic variants, material mismatches, missing fields, incorrect warnings, low contrast, blur, glare, perspective skew, rotation, and cropped content. Do not tune against the same images used for final performance evaluation.

## 🔒 Secure Development Requirements

* Pin dependencies through the repository's lock mechanism and scan them for known vulnerabilities.
* Run secret scanning before every merge.
* Validate file signatures, dimensions, decompression ratio, and decoded pixel count.
* Reject archive traversal, nested archives, and encrypted archives unless explicitly supported.
* Use random server-side temporary names and never trust a client filename as a path.
* Apply least-privilege managed identities in Azure.
* Disable request-body, image, and OCR-text logging.
* Sanitize errors returned to clients while retaining a correlation ID.
* Threat-model upload abuse, denial of service, formula injection, authorization bypass, and sensitive-data disclosure.
* Document and test temporary-data disposal.

## 📈 Logging and Telemetry

Structured logs may contain:

* Correlation ID.
* Route and response status.
* Component durations.
* File size and media type.
* Count of batch items.
* Technical error code.
* Application version and OCR model version.

Logs must not contain images, OCR text, application field values, addresses, authorization headers, access keys, or full client filenames. Metrics should cover request volume, duration percentiles, dependency failures, manual-review rate, and technical error rate.

## ☁️ Azure Deployment

The deployment pipeline should:

1. Build from a pinned runtime base image.
2. Run formatting, static analysis, tests, dependency scanning, and secret scanning.
3. Produce a versioned, immutable artifact.
4. Deploy infrastructure through reviewed infrastructure-as-code.
5. Assign managed identities and least-privilege roles.
6. Apply HTTPS-only, network, and temporary-storage controls.
7. Run smoke and readiness tests.
8. Verify that the target agency network can reach every required endpoint.
9. Run a representative performance check.
10. Record the deployed application and rules versions.

Use deployment slots or an equivalent staged rollout where available. Roll back by redeploying the preceding immutable artifact; do not repair production instances manually.

## 🧹 Data Disposal

Temporary content must have both normal-path and failure-path cleanup. Implement:

* Immediate cleanup after synchronous responses.
* Cleanup after batch completion, cancellation, and expiration.
* A scheduled reaper for abandoned artifacts.
* A maximum retention configuration.
* Metrics for expired-item cleanup failures without logging content.
* Integration tests that simulate exceptions at every processing stage.

## 📝 Documentation Standards

Every public API route, domain model, configuration setting, comparison rule, and operational dependency must be documented. Code comments should explain purpose and contract, not narrate obvious statements. Documentation examples must use synthetic data.

Before merging a documentation change:

```text
mkdocs build --strict
```

If the repository uses another documentation command, publish that verified command instead. Strict builds should fail on unresolved links and warnings.

## ✅ Definition of Done

A change is complete only when:

* The implementation satisfies an identified requirement or defect.
* Input, output, state, and error contracts are explicit.
* Automated tests cover normal, boundary, and failure behavior.
* No sensitive fixture or credential is introduced.
* Accessibility impact has been evaluated.
* Performance impact has been measured when the processing path changes.
* API and user documentation are updated.
* Temporary-data cleanup remains verified.
* The documentation site builds without warnings.
* The complete diff has been reviewed for unrelated changes.

## 🔗 Related Documentation

* [Documentation Home](index.md)
* [System Architecture](architecture.md)
* [User Guide](user-guide/index.md)
* [API Reference](api/index.md)

