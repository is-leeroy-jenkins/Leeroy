# Prompt Engineering

Prompt Engineering mode provides a SQLite-backed interface for managing reusable prompt templates.

## 🧭 Purpose

Prompt Engineering mode lets users create, search, sort, page, select, edit, update, and delete
prompt records stored in the local SQLite `Prompts` table. These templates can be reused as system
instructions for Text Generation and Document Q&A workflows.

## 🧱 Workflow Position

```text

Create or Select Prompt Template
  │
  ▼
Store Prompt in SQLite
  │
  ▼
Load Prompt by Caption
  │
  ▼
Cascade into System Instructions
  │
  ▼
Use in Text Generation or Document Q&A

```

## 🗄️ Prompt Table

Prompt templates are stored in the SQLite `Prompts` table.

| Field       | Purpose                                       |
| ----------- | --------------------------------------------- |
| `PromptsId` | Primary key for each prompt record.           |
| `Caption`   | Display label used in template selectors.     |
| `Name`      | Prompt name or internal title.                |
| `Text`      | Full prompt content.                          |
| `Version`   | Version label or revision identifier.         |
| `ID`        | Optional external or user-defined identifier. |

## 🔍 Searching and Sorting

Prompt Engineering mode supports search, sorting, and paging.

| Control       | Purpose                                      |
| ------------- | -------------------------------------------- |
| Search        | Filters prompts by name or text.             |
| Sort by       | Selects the field used for ordering rows.    |
| Direction     | Controls ascending or descending sort order. |
| Go to ID      | Jumps directly to a prompt by identifier.    |
| Previous/Next | Moves through paged prompt rows.             |

## 🧾 Selecting a Prompt

Select a prompt row from the table to load it into the edit fields.

Only one prompt should be selected at a time. If multiple rows are selected, the application warns
the user to select exactly one prompt.

Loaded prompt fields populate:

| Field   | Use                  |
| ------- | -------------------- |
| Caption | Display caption.     |
| Name    | Prompt name.         |
| Text    | Prompt body.         |
| Version | Revision label.      |
| ID      | Optional identifier. |

## 🖊️ Editing Prompts

Use the Edit Prompt expander to update prompt metadata and text.

Common edits include:

| Edit             | Reason                               |
| ---------------- | ------------------------------------ |
| Revise `Text`    | Improve behavior or add constraints. |
| Update `Version` | Track prompt changes.                |
| Change `Caption` | Make the prompt easier to select.    |
| Change `Name`    | Improve prompt organization.         |
| Clear selection  | Prepare to create a new prompt.      |

## ➕ Creating a Prompt

To create a prompt:

1. Clear the current selection.
2. Enter a caption.
3. Enter a name.
4. Enter prompt text.
5. Enter a version or identifier if useful.
6. Save the prompt.

Example prompt text:

```text

Respond as a concise federal data analyst. Use a short executive summary, followed by key findings, risks, and recommended next steps.

```

## 🔁 Cascading into System Instructions

Prompt Engineering mode can cascade a selected prompt into the System Instructions field.

Use this when a prompt template should control the next chat or document-answering workflow.

Recommended use cases:

| Use Case         | Example                                                                      |
| ---------------- | ---------------------------------------------------------------------------- |
| Federal analysis | Budget, audit, policy, acquisition, or program-analysis framing.             |
| Documentation    | MkDocs, Google-style docstring, README, or API-reference instructions.       |
| Data analysis    | Summary, anomaly review, feature interpretation, or model-comparison format. |
| Drafting         | Memo, cover letter, response, recommendation, or technical explanation.      |

## 🧪 Example Prompt Template

```text 

Caption: Federal Analytical Summary
Name: federal_summary
Version: 1.0
ID: FED-SUMMARY-001

Text:
You are supporting a federal analyst. Provide a concise summary using:
1. Purpose
2. Key facts
3. Risks
4. Constraints
5. Recommended next steps

Use plain language and avoid unsupported assumptions.

```

## ✅ Recommended Sequence

1. Create reusable prompts for repeat workflows.
2. Use clear captions because captions appear in selectors.
3. Version prompts when changing behavior.
4. Keep prompts specific enough to control output format.
5. Cascade selected prompts into system instructions when needed.
6. Test prompts in Text Generation before using them for high-value workflows.

## 🧯 Troubleshooting

| Issue                              | Likely Cause                                      | Fix                                                      |
| ---------------------------------- | ------------------------------------------------- | -------------------------------------------------------- |
| Prompt does not appear in selector | Caption may be blank or database did not refresh. | Confirm `Caption` is populated and reload the app.       |
| Multiple prompts selected          | More than one table row has `Selected` checked.   | Select exactly one prompt.                               |
| Prompt changes not reflected       | The active system instructions were not updated.  | Cascade or reload the template into system instructions. |
| Prompt table is empty              | No records exist in `Prompts`.                    | Create a new prompt record.                              |

## 🔗 Related API Pages

| API Page                              | Purpose                                                                                             |
| ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| [App API](../api/app.md)              | Source documentation for prompt lookup, selection, creation, update, deletion, and prompt building. |
| [Configuration API](../api/config.md) | UI help text and database path configuration.                                                       |
