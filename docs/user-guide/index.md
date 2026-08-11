# 👤 User Guide

## 📋 Purpose

The AI-Powered Alcohol Label Verification prototype helps a compliance agent compare information visible on an alcohol beverage label with corresponding application data. It reduces repetitive matching work while keeping the agent responsible for the final compliance decision.

The prototype does not approve a label, replace regulatory judgment, or write information to COLA.

## 🚀 Quick Start

Complete a single-label review in three actions:

1. Enter or paste the application values.
2. Upload the label image.
3. Select **Verify Label**.

The result shows the expected application value, the value detected on the label, the comparison status, severity, OCR confidence, and supporting evidence.

## 📥 Supported Inputs

The interface should identify the formats enabled by the deployed build. A typical prototype configuration supports JPEG and PNG images and may support PDF label artwork. Batch mode may accept multiple selected files or a ZIP archive when ZIP processing is enabled.

For the best results:

* Use the original digital artwork when available.
* Include the complete label rather than a tightly cropped field.
* Keep text upright, in focus, and large enough to read.
* Avoid reflections, glare, fingers, and background clutter.
* Use one label application reference per uploaded item.

The system attempts to handle skew, low contrast, and moderate glare, but it must request manual review when the image does not provide dependable evidence.

## 📝 Application Fields

Enter the values exactly as they appear in the application. Depending on beverage type, the review may include:

| Field | Example | Notes |
|---|---|---|
| Brand name | `OLD TOM DISTILLERY` | Case and minor punctuation differences may be acceptable |
| Class/type | `Kentucky Straight Bourbon Whiskey` | Use the full application designation |
| Alcohol content | `45% Alc./Vol. (90 Proof)` | Include ABV and proof when present |
| Net contents | `750 mL` | Include the unit |
| Producer/bottler | Name and address | Enter the complete application value |
| Country of origin | `United States` | Primarily applicable to imported products |
| Government warning | Prescribed warning statement | Evaluated under strict text and formatting rules |

Required information varies among distilled spirits, wine, and malt beverages. TTB publishes product-specific labeling guidance and Beverage Alcohol Manuals through its [labeling resources](https://www.ttb.gov/regulated-commodities/labeling/labeling-resources).

## 🖼️ Single-Label Review

1. Open the label verification page.
2. Select the beverage type.
3. Enter the application reference if the interface provides one. Use a non-sensitive reference for prototype testing.
4. Complete the applicable application fields.
5. Choose or drag the label image into the upload area.
6. Confirm the image preview is complete and readable.
7. Select **Verify Label** once.
8. Follow the progress indicator until the results appear.
9. Review every failed, missing, low-confidence, or manual-review field.
10. Download the result if a record is needed outside the prototype.

## 📦 Batch Review

Batch processing is intended for several independent label/application pairs. The initial prototype target is 20–50 images, even though future operational workloads may be larger.

1. Prepare files using the naming or manifest convention shown by the deployed interface.
2. Confirm that each application row maps to exactly one label image.
3. Upload the files or supported archive.
4. Resolve duplicate, missing, or unmapped references before processing.
5. Select **Verify Batch**.
6. Monitor completed, remaining, and failed counts.
7. Open any failed item to see its correction instructions.
8. Review all critical and manual-review findings.
9. Download the CSV summary.

A failed image must not erase or invalidate successful results from other items.

## 📊 Understanding Results

### Status

| Status | Meaning | Agent action |
|---|---|---|
| Pass | The available evidence satisfies the configured comparison rule | Confirm the evidence as part of the normal review |
| Fail | The observed value conflicts with the expected value or required rule | Examine the evidence and determine the compliance action |
| Manual review | The system cannot reach a dependable automated conclusion | Inspect the image and application directly |
| Not found | The field could not be located on the submitted label image | Check whether it is absent, obscured, or outside the image |
| Not applicable | The field does not apply to the selected beverage or application | No action unless the beverage type is incorrect |

### Severity

* **Critical** identifies a failure with substantial compliance significance, such as an absent or incorrect government warning.
* **Major** identifies a material field mismatch that needs agent attention.
* **Minor** identifies a variation likely to be acceptable but worth confirming.
* **Info** provides context without asserting a compliance failure.

### Confidence

Confidence describes how certain the extraction system is that it read the text correctly. It does not measure legal compliance. A high-confidence OCR value can still fail a compliance rule, and a low-confidence match still requires manual review.

## ⚖️ Matching Behavior

### Fuzzy matching

Brand names and similar descriptive fields may ignore harmless differences in capitalization, spacing, punctuation, and typographic apostrophes. For example, `STONE'S THROW` and `Stone’s Throw` can be treated as equivalent while the result continues to show both original values.

Fuzzy matching must not be used to conceal missing words, changed quantities, changed addresses, or materially different class/type designations.

### Numeric matching

The system extracts a number and unit before comparing alcohol content and net contents. It may compare equivalent unit representations, but it must not treat a materially different quantity as a formatting difference.

### Government health warning

The government warning receives strict review. TTB requires `GOVERNMENT WARNING` in capital letters and bold type, with the prescribed statement presented continuously and separately from other information. See the current [TTB health-warning guidance](https://www.ttb.gov/regulated-commodities/beverage-alcohol/distilled-spirits/ds-labeling-home/ds-health-warning).

The system should require manual review when the image does not reliably show capitalization, bold type, continuity, contrast, placement, or size. A matching OCR string alone is not sufficient evidence of correct formatting.

## ♿ Accessibility Features

The deployed interface should provide:

* Full keyboard operation.
* A visible focus indicator.
* High-contrast mode.
* Large-text support without clipped controls.
* Status icons and text in addition to color.
* Progress and error messages readable by assistive technology.
* Tooltips or help text explaining comparisons.

If an interaction cannot be completed using the keyboard, record the browser, page, and affected control for the development team.

## 🛠️ Correcting Common Problems

| Problem | Resolution |
|---|---|
| File type is not accepted | Convert the image to an enabled format without reducing readability |
| Image is blurred | Upload the original artwork or take a new, steady, well-lit photograph |
| Image contains glare | Re-photograph the container with indirect lighting and a changed angle |
| Text is skewed | Retake the image face-on or upload the original flat label |
| Required field is not found | Confirm that the complete front, back, and side label content was submitted |
| Values were paired with the wrong image | Correct the application/image mapping and run that item again |
| Processing exceeds the expected time | Wait for the current request; retry only after an error is shown |
| Service is unavailable | Retain the request ID shown in the error and contact support |
| Result appears wrong | Use the displayed evidence, perform a manual review, and report the case without sensitive content |

## 🔐 Privacy and Data Handling

The prototype is intended for test material and is not authorized merely by its existence to process production data. Do not upload sensitive or personally identifiable information unless the deployment owner has explicitly approved that use.

Uploaded images and extracted values should be held only long enough to complete processing and return the result. Download any permitted result before leaving the page because the prototype may not retain it.

## ✅ Review Checklist

Before relying on a result, confirm that:

* The image belongs to the intended application.
* The image includes every relevant label panel.
* The application values were entered correctly.
* Each field has been reviewed, not only the overall status.
* Critical, failed, missing, and manual-review findings were examined.
* The government warning was visually inspected when formatting evidence is uncertain.
* The final compliance decision was made by an authorized reviewer.

## 🔗 Related Documentation

* [Documentation Home](../index.md)
* [System Architecture](../architecture.md)
* [API Reference](../api/index.md)
* [Development Guide](../development.md)

