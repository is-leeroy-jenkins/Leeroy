# Data Management

Data Management mode provides a SQLite-backed interface for importing, browsing, profiling,
filtering, aggregating, visualizing, and administering local data tables.

## 🧭 Purpose

Data Management mode gives Leeroy a practical analytical workspace over SQLite. Users can inspect
existing tables, import tabular data, run simple data-quality checks, create summaries, visualize
values, administer schema changes, and run guarded read-only SQL queries.

## 🧱 Workflow Position

```text id="9lj02u"
SQLite Database
  │
  ├── Imported Tables
  ├── Prompt Tables
  ├── Chat History
  └── Embedding Tables
  │
  ▼
Data Management Mode
  │
  ├── Browse
  ├── CRUD
  ├── Explore
  ├── Filter
  ├── Aggregate
  ├── Visualize
  ├── Admin
  └── SQL
  │
  ▼
Interactive Analysis and Administration
```

## 🗄️ Database Connection

Leeroy uses the configured SQLite database path for application data. Data Management utilities open
bounded SQLite connections, perform the requested operation, and close the connection through
context managers.

Core database helpers include:

| Helper                | Purpose                                       |
| --------------------- | --------------------------------------------- |
| `create_connection()` | Opens a SQLite connection.                    |
| `list_tables()`       | Lists existing tables.                        |
| `create_schema()`     | Reads table schema metadata.                  |
| `read_table()`        | Reads table rows into a pandas DataFrame.     |
| `drop_table()`        | Drops a selected table.                       |
| `create_index()`      | Creates an index on a validated table column. |

## 📥 Import Workflow

Use import tools to bring tabular data into SQLite.

Recommended import sequence:

1. Select the import tab or import area.
2. Upload a supported tabular source.
3. Review detected sheets or tables.
4. Convert the DataFrame schema into SQLite column definitions.
5. Insert rows into the selected table.
6. Browse the table to confirm import success.

When converting pandas data types to SQLite types, Leeroy maps common types to `INTEGER`, `REAL`, or
`TEXT`.

## 🔎 Browse and Explore

Browsing and exploration tools let users inspect table contents without writing SQL manually.

| Feature        | Purpose                                                     |
| -------------- | ----------------------------------------------------------- |
| Table selector | Chooses an existing SQLite table.                           |
| Schema display | Shows column names, types, nullability, defaults, and keys. |
| Row display    | Shows data in a DataFrame-style grid.                       |
| Pagination     | Supports viewing subsets of large tables.                   |
| Row count      | Confirms table size.                                        |

## 🧹 CRUD-Style Operations

Data Management mode can support creating, updating, and deleting row-level data depending on the
active UI controls.

Use CRUD features carefully. These actions change the local SQLite database.

Recommended practice:

| Practice                      | Reason                                            |
| ----------------------------- | ------------------------------------------------- |
| Inspect schema before editing | Confirms column names and types.                  |
| Edit small batches            | Reduces accidental changes.                       |
| Confirm row identifiers       | Prevents modifying the wrong record.              |
| Back up important databases   | SQLite files are easy to copy before major edits. |

## 🧪 Data Profiling

Profiling calculates basic data-quality and distribution metrics.

Typical profile outputs include:

| Metric              | Description                        |
| ------------------- | ---------------------------------- |
| Column              | Column name.                       |
| Data type           | pandas dtype for the column.       |
| Null percentage     | Share of rows with missing values. |
| Distinct percentage | Share of unique values.            |
| Minimum             | Minimum value for numeric columns. |
| Maximum             | Maximum value for numeric columns. |
| Mean                | Average value for numeric columns. |

Use profiling to identify missingness, low-cardinality fields, numeric ranges, and potential
data-quality issues.

## 🧰 Filtering

The filter workflow allows users to select a column, choose an operator, enter a value, and return
matching rows.

Supported operators include:

| Operator   | Meaning                         |
| ---------- | ------------------------------- |
| `=`        | Equal to value.                 |
| `!=`       | Not equal to value.             |
| `>`        | Greater than value.             |
| `<`        | Less than value.                |
| `>=`       | Greater than or equal to value. |
| `<=`       | Less than or equal to value.    |
| `contains` | Text contains value.            |

## 🧮 Aggregation

Aggregation tools summarize numeric columns.

Supported aggregations include:

| Aggregation | Description                |
| ----------- | -------------------------- |
| `COUNT`     | Count non-null values.     |
| `SUM`       | Sum values.                |
| `AVG`       | Calculate arithmetic mean. |
| `MIN`       | Find minimum value.        |
| `MAX`       | Find maximum value.        |
| `MEDIAN`    | Calculate median value.    |

## 📊 Visualization

Leeroy uses Plotly Express for interactive charts.

Supported chart types include:

| Chart       | Use                                         |
| ----------- | ------------------------------------------- |
| Histogram   | Numeric distribution.                       |
| Bar         | Category-to-value comparison.               |
| Line        | Ordered or time-like trend.                 |
| Scatter     | Relationship between two numeric variables. |
| Box         | Distribution and outlier review.            |
| Pie         | Category share.                             |
| Correlation | Numeric correlation matrix.                 |

## 🛠️ Schema Administration

Schema tools support controlled table administration.

| Action        | Purpose                                                                    |
| ------------- | -------------------------------------------------------------------------- |
| Create table  | Builds a custom SQLite table from column definitions.                      |
| Add column    | Adds a validated column to an existing table.                              |
| Rename column | Renames an existing column with native SQLite support or fallback rebuild. |
| Rename table  | Renames a table with native SQLite support or fallback rebuild.            |
| Drop column   | Rebuilds the table without the selected column when safe.                  |
| Create index  | Adds an index for a validated table column.                                |
| Drop table    | Removes a selected table.                                                  |

## 🧯 Guarded SQL

The SQL console allows read-only SQL workflows and blocks destructive operations.

Allowed starting patterns include:

| Pattern   | Purpose                                   |
| --------- | ----------------------------------------- |
| `SELECT`  | Query rows and columns.                   |
| `WITH`    | Use a common table expression.            |
| `EXPLAIN` | Inspect query plans.                      |
| `PRAGMA`  | Run read-oriented SQLite metadata checks. |

Blocked operations include:

```text id="9aouol"
INSERT
UPDATE
DELETE
DROP
ALTER
CREATE
ATTACH
DETACH
VACUUM
REPLACE
TRIGGER
```

This keeps SQL exploration safer while still allowing useful inspection and analysis.

## 🧪 Example Queries

List tables:

```sql id="eph1r5"
SELECT name
FROM sqlite_master
WHERE type = 'table'
ORDER BY name;
```

Preview prompt records:

```sql id="g3xtgk"
SELECT PromptsId, Caption, Name, Version
FROM Prompts
ORDER BY PromptsId DESC
LIMIT 20;
```

Count chat messages by role:

```sql id="eklybg"
SELECT role, COUNT(*) AS message_count
FROM chat_history
GROUP BY role;
```

## ✅ Recommended Sequence

1. Browse tables before making changes.
2. Inspect schema before importing or editing data.
3. Profile tables to identify data-quality issues.
4. Filter and aggregate before visualizing.
5. Use guarded SQL for custom read-only analysis.
6. Back up the SQLite database before major schema administration.

## 🔗 Related API Pages

| API Page                              | Purpose                                                                                                                       |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [App API](../api/app.md)              | Source documentation for SQLite connections, schema tools, filtering, aggregation, visualization, profiling, and guarded SQL. |
| [Configuration API](../api/config.md) | Database path and runtime constants.                                                                                          |
