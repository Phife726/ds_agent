---
name: ds-data-engineering
description: |
  Builds and maintains the data infrastructure that DS projects run on — pipelines, ETL/ELT jobs, data warehouses, orchestration, and data quality systems. Use this skill whenever the project requires building the systems that create and move data, not just consuming data that already exists. Trigger when the user says "build a data pipeline", "ETL", "ELT", "data ingestion", "load data into a warehouse", "orchestrate my pipelines", "Airflow", "Prefect", "dbt", "data model", "incremental load", "CDC (change data capture)", "data quality checks", "schema evolution", "data lake vs warehouse", "partition my data", "slowly changing dimensions", "data freshness", "pipeline is failing", "automate my data jobs", "build a feature store", or anything about creating infrastructure for data rather than analyzing it. Also trigger when a DS project keeps failing because the data feeding it is unreliable, late, or malformed — the fix is usually infrastructure, not modeling.
---

# Data Engineering Skill

This skill builds the infrastructure layer that everything else depends on. Data scientists often inherit data pipelines; data engineers build them. But in many teams, one person does both — and even pure DS work sometimes requires spinning up a pipeline to get the right data in the right place.

## The Data Engineering Layer in a DS Stack

```
┌─────────────────────────────────────────────────────┐
│             Data Science & ML Layer                  │
│   (ds-eda-process, ds-supervised-modeling, etc.)    │
├─────────────────────────────────────────────────────┤
│              Feature Store / Analytics Layer         │
│   (ds-feature-engineering, data-analyst)            │
├─────────────────────────────────────────────────────┤
│          DATA ENGINEERING LAYER (This Skill)         │
│   ETL/ELT pipelines, warehouse, data quality        │
├─────────────────────────────────────────────────────┤
│              Raw Data Sources                        │
│   APIs, databases, files, streams, SaaS tools       │
└─────────────────────────────────────────────────────┘
```

---

## Architecture Decision: ETL vs ELT, Lake vs Warehouse

Before writing any code, help the user answer: what are we building, and why?

### ETL vs ELT

| | ETL (Transform before load) | ELT (Transform after load) |
|---|---|---|
| **Where** | Transform in pipeline code | Transform in warehouse using SQL |
| **When to use** | Sensitive data that must be anonymized before storage; legacy warehouses | Modern cloud warehouses (BigQuery, Snowflake, Redshift) |
| **Tradeoffs** | More control; harder to re-run; transform code in Python | Easier to re-run; SQL is accessible; full raw data retained |

**The modern default is ELT**: load raw data into the warehouse first, then transform it using SQL (with dbt). This keeps the raw data as the source of truth and lets you re-run transformations easily.

### Data Lake vs Data Warehouse vs Lakehouse

| | Data Lake | Data Warehouse | Lakehouse |
|---|---|---|---|
| **Storage** | Files (Parquet, JSON, CSV) in object storage | Columnar, structured tables | Files + metadata layer (Delta, Iceberg) |
| **Schema** | Schema-on-read (flexible) | Schema-on-write (strict) | Both |
| **Cost** | Very low storage | Higher compute | Moderate |
| **Best for** | Raw data, ML training data | Business analytics, reporting | Both at once |
| **Examples** | S3, GCS | BigQuery, Snowflake, Redshift | Databricks, Delta Lake |

**For most DS teams**: Start with a warehouse (BigQuery, Snowflake, or local DuckDB for small projects). Add a data lake if you need ML training data or cheap long-term storage.

---

## Building ETL Pipelines in Python

### The Foundation: A Well-Structured Pipeline

```python
import pandas as pd
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineRun:
    """Tracks a single pipeline execution for auditing."""
    pipeline_name: str
    start_time: datetime = None
    end_time: datetime = None
    rows_processed: int = 0
    status: str = 'pending'
    error_message: Optional[str] = None

    def start(self):
        self.start_time = datetime.utcnow()
        self.status = 'running'
        logger.info(f"Pipeline '{self.pipeline_name}' started")

    def complete(self, rows: int):
        self.end_time = datetime.utcnow()
        self.rows_processed = rows
        self.status = 'success'
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"Pipeline '{self.pipeline_name}' completed: {rows:,} rows in {duration:.1f}s")

    def fail(self, error: Exception):
        self.end_time = datetime.utcnow()
        self.status = 'failed'
        self.error_message = str(error)
        logger.error(f"Pipeline '{self.pipeline_name}' FAILED: {error}")


def extract_from_api(url: str, params: dict) -> pd.DataFrame:
    """Extract data from a REST API."""
    import requests
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def extract_from_database(query: str, connection_string: str) -> pd.DataFrame:
    """Extract data from a SQL database."""
    import sqlalchemy
    engine = sqlalchemy.create_engine(connection_string)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def load_to_database(df: pd.DataFrame, table_name: str, connection_string: str,
                     if_exists: str = 'append') -> int:
    """Load a DataFrame to a SQL table. Returns rows written."""
    import sqlalchemy
    engine = sqlalchemy.create_engine(connection_string)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False, method='multi', chunksize=1000)
    return len(df)


def run_pipeline(extract_fn, transform_fn, load_fn, pipeline_name: str) -> PipelineRun:
    """
    Generic pipeline runner with error handling and audit logging.
    Separation of Extract, Transform, Load makes each step testable independently.
    """
    run = PipelineRun(pipeline_name=pipeline_name)
    run.start()
    try:
        raw_data = extract_fn()
        transformed = transform_fn(raw_data)
        rows = load_fn(transformed)
        run.complete(rows)
    except Exception as e:
        run.fail(e)
        raise
    return run
```

### Incremental Loading (The Key to Efficient Pipelines)

Never reload all data from scratch every run unless you have to. Full refreshes are slow, expensive, and prone to timeouts.

```python
from datetime import datetime, timedelta
import sqlalchemy as sa

def get_last_loaded_timestamp(engine, table_name: str, timestamp_col: str) -> datetime:
    """Find the most recent record in the target table to use as incremental boundary."""
    with engine.connect() as conn:
        result = conn.execute(
            sa.text(f"SELECT MAX({timestamp_col}) FROM {table_name}")
        ).scalar()
    if result is None:
        return datetime(2000, 1, 1)  # Full load on first run
    return result


def incremental_extract(
    engine,
    source_table: str,
    timestamp_col: str,
    last_loaded: datetime,
    buffer_hours: int = 1  # Overlap buffer handles late-arriving records
) -> pd.DataFrame:
    """
    Extract only records newer than last_loaded, with a safety buffer.
    The buffer prevents missing records that arrived slightly late.
    """
    boundary = last_loaded - timedelta(hours=buffer_hours)
    query = f"""
        SELECT *
        FROM {source_table}
        WHERE {timestamp_col} > :boundary
        ORDER BY {timestamp_col}
    """
    with engine.connect() as conn:
        return pd.read_sql(sa.text(query), conn, params={'boundary': boundary})


def upsert_records(df: pd.DataFrame, table_name: str, engine, primary_key: str) -> int:
    """
    Insert new records, update existing ones (upsert).
    Handles duplicate records from the overlap buffer.
    """
    # For PostgreSQL, use ON CONFLICT DO UPDATE
    # For SQLite, use INSERT OR REPLACE INTO
    # This example uses the staging table pattern (works everywhere)

    staging_table = f"{table_name}_staging"
    df.to_sql(staging_table, engine, if_exists='replace', index=False)

    with engine.begin() as conn:
        # Delete existing records that will be updated
        conn.execute(sa.text(f"""
            DELETE FROM {table_name}
            WHERE {primary_key} IN (SELECT {primary_key} FROM {staging_table})
        """))
        # Insert all staging records
        conn.execute(sa.text(f"""
            INSERT INTO {table_name} SELECT * FROM {staging_table}
        """))
        conn.execute(sa.text(f"DROP TABLE {staging_table}"))

    return len(df)
```

---

## Data Quality and Validation

Garbage in, garbage out. Data pipelines without quality checks will silently corrupt downstream models and reports.

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class QualityCheck:
    name: str
    passed: bool
    message: str


def validate_dataframe(df: pd.DataFrame, rules: dict) -> list[QualityCheck]:
    """
    Run a set of data quality checks on a DataFrame.
    Rules structure:
    {
        'row_count': {'min': 100, 'max': 1_000_000},
        'null_thresholds': {'revenue': 0.01, 'customer_id': 0.0},
        'value_ranges': {'age': (0, 120), 'revenue': (0, None)},
        'unique_keys': ['transaction_id'],
        'allowed_values': {'status': ['active', 'inactive', 'pending']},
    }
    """
    results = []

    if 'row_count' in rules:
        rule = rules['row_count']
        min_ok = len(df) >= rule.get('min', 0)
        max_ok = len(df) <= rule.get('max', float('inf'))
        results.append(QualityCheck(
            'row_count',
            min_ok and max_ok,
            f"Row count: {len(df):,} (expected {rule.get('min', '?')} - {rule.get('max', '?')})"
        ))

    if 'null_thresholds' in rules:
        for col, max_null_rate in rules['null_thresholds'].items():
            if col in df.columns:
                null_rate = df[col].isnull().mean()
                results.append(QualityCheck(
                    f'null_rate:{col}',
                    null_rate <= max_null_rate,
                    f"Null rate for '{col}': {null_rate:.2%} (max allowed: {max_null_rate:.2%})"
                ))

    if 'value_ranges' in rules:
        for col, (min_val, max_val) in rules['value_ranges'].items():
            if col in df.columns:
                if min_val is not None and (df[col] < min_val).any():
                    results.append(QualityCheck(
                        f'range:{col}',
                        False,
                        f"'{col}' has {(df[col] < min_val).sum()} values below minimum {min_val}"
                    ))
                elif max_val is not None and (df[col] > max_val).any():
                    results.append(QualityCheck(
                        f'range:{col}',
                        False,
                        f"'{col}' has {(df[col] > max_val).sum()} values above maximum {max_val}"
                    ))
                else:
                    results.append(QualityCheck(f'range:{col}', True, f"'{col}' within range"))

    if 'unique_keys' in rules:
        for col in rules['unique_keys']:
            if col in df.columns:
                dupes = df[col].duplicated().sum()
                results.append(QualityCheck(
                    f'unique:{col}',
                    dupes == 0,
                    f"'{col}' has {dupes} duplicate values"
                ))

    # Print summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"Data Quality: {passed}/{len(results)} checks passed, {failed} failed")
    for r in results:
        status = '✓' if r.passed else '✗'
        print(f"  {status} {r.name}: {r.message}")

    return results
```

**Where to run quality checks**: Before loading to the destination table (fail fast), and after loading (confirm the data arrived correctly). The first catches bad source data; the second catches pipeline bugs.

---

## Orchestration with Apache Airflow

For pipelines that need to run on a schedule, depend on each other, and be monitored, use an orchestrator. Airflow is the industry standard.

```python
# dags/customer_features_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

# Define default arguments that apply to all tasks
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,        # Don't wait for previous day's run to succeed
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email': ['alerts@yourcompany.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='customer_features_daily',
    default_args=default_args,
    description='Daily customer feature table refresh',
    schedule_interval='0 4 * * *',  # 4am daily
    catchup=False,                   # Don't backfill missed runs
    tags=['features', 'customers'],
) as dag:

    extract_transactions = PythonOperator(
        task_id='extract_transactions',
        python_callable=run_incremental_extract,
        op_kwargs={'table': 'transactions', 'lookback_hours': 25}
    )

    run_quality_checks = PythonOperator(
        task_id='validate_raw_data',
        python_callable=validate_pipeline_data,
    )

    transform_features = BashOperator(
        task_id='dbt_run_features',
        bash_command='cd /opt/dbt && dbt run --select customers.features'
    )

    test_features = BashOperator(
        task_id='dbt_test_features',
        bash_command='cd /opt/dbt && dbt test --select customers.features'
    )

    # Set dependencies: extract → validate → transform → test
    extract_transactions >> run_quality_checks >> transform_features >> test_features
```

**DAG design principles**:
- Each task should be idempotent — running it twice should produce the same result as running it once.
- Tasks should be small and focused. A 500-line Python function is a single DAG task that's impossible to debug; break it into 5 tasks.
- Use `catchup=False` unless you specifically need historical backfilling.
- Always add retry logic and alerting to production DAGs.

---

## Data Transformation with dbt

dbt (data build tool) is the standard for SQL-based transformations in a warehouse. It turns raw loaded data into clean, tested, documented analytical tables.

### dbt Project Structure

```
dbt_project/
├── models/
│   ├── staging/          # 1:1 with source tables, light cleaning only
│   │   └── stg_transactions.sql
│   ├── intermediate/     # Joins, complex business logic
│   │   └── int_customer_orders.sql
│   └── marts/            # Final analytical tables consumed by analysts & models
│       └── customers.sql
├── tests/                # Data quality tests
├── sources.yml           # Declare raw source tables
└── dbt_project.yml       # Project config
```

```sql
-- models/staging/stg_transactions.sql
-- Staging layer: clean names, cast types, no business logic
WITH source AS (
    SELECT * FROM {{ source('raw', 'transactions') }}
),

renamed AS (
    SELECT
        transaction_id,
        customer_id,
        CAST(amount AS NUMERIC) AS amount_usd,
        LOWER(status) AS status,
        CAST(created_at AS TIMESTAMP) AS transaction_at,
        -- Standardize: remove whitespace from string fields
        TRIM(product_category) AS product_category
    FROM source
    WHERE transaction_id IS NOT NULL  -- Remove rows with null PKs at source
)

SELECT * FROM renamed
```

```sql
-- models/marts/customers.sql
-- Mart layer: business-ready customer table with aggregated features
{{
  config(
    materialized='table',    -- Persisted as a real table
    partition_by={'field': 'first_purchase_date', 'data_type': 'date'},
    cluster_by=['customer_segment']
  )
}}

WITH orders AS (
    SELECT * FROM {{ ref('int_customer_orders') }}
),

features AS (
    SELECT
        customer_id,
        COUNT(*) AS n_orders,
        SUM(amount_usd) AS total_revenue,
        AVG(amount_usd) AS avg_order_value,
        MIN(order_at) AS first_purchase_date,
        MAX(order_at) AS last_purchase_date,
        DATE_DIFF(CURRENT_DATE, MAX(DATE(order_at)), DAY) AS days_since_last_order
    FROM orders
    GROUP BY customer_id
)

SELECT * FROM features
```

### dbt Tests for Data Quality

```yaml
# models/marts/schema.yml
version: 2

models:
  - name: customers
    description: "One row per customer with aggregated purchase behavior"
    columns:
      - name: customer_id
        tests:
          - unique           # No duplicate customer IDs
          - not_null         # Every row has a customer ID
      - name: total_revenue
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0"  # Revenue can't be negative
      - name: days_since_last_order
        tests:
          - dbt_utils.expression_is_true:
              expression: ">= 0"
```

Run `dbt test` to validate all assertions after every transformation run.

---

## Working with Files: Parquet and Data Lakes

For data pipelines that don't use a SQL warehouse, Parquet files on object storage are the standard.

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def write_partitioned_parquet(
    df: pd.DataFrame,
    base_path: str,
    partition_cols: list[str]
) -> None:
    """
    Write DataFrame as a Hive-style partitioned Parquet dataset.
    Partitioning by date allows downstream queries to skip files they don't need.
    """
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(
        table,
        root_path=base_path,
        partition_cols=partition_cols,
        compression='snappy',       # Snappy is fast; use 'zstd' for better compression
        existing_data_behavior='overwrite_or_ignore'
    )
    print(f"Written to {base_path} partitioned by {partition_cols}")


def read_partitioned_parquet(
    base_path: str,
    filters: list = None
) -> pd.DataFrame:
    """
    Read partitioned Parquet with optional partition pruning.
    Filters like [('year', '=', '2024'), ('month', '>=', '6')] only read matching partitions.
    """
    dataset = pq.ParquetDataset(base_path, filters=filters)
    return dataset.read_pandas().to_pandas()

# Example: write daily transaction data with date partitioning
transactions_df['year'] = pd.to_datetime(transactions_df['transaction_date']).dt.year.astype(str)
transactions_df['month'] = pd.to_datetime(transactions_df['transaction_date']).dt.month.astype(str).str.zfill(2)

write_partitioned_parquet(
    transactions_df,
    base_path='s3://my-data-lake/transactions/',
    partition_cols=['year', 'month']
)
```

### DuckDB for Local Querying

DuckDB is an embedded analytical database — you can run SQL on Parquet files, CSVs, and DataFrames without setting up any infrastructure. It's the best tool for small-to-medium data engineering work.

```python
import duckdb

# DuckDB can query Parquet files directly with SQL
conn = duckdb.connect()

result = conn.execute("""
    SELECT
        customer_id,
        SUM(amount) AS total_revenue,
        COUNT(*) AS n_transactions,
        MAX(transaction_date) AS last_transaction
    FROM read_parquet('data/transactions/*.parquet')
    WHERE transaction_date >= '2024-01-01'
    GROUP BY customer_id
    ORDER BY total_revenue DESC
""").df()

# Write results back to Parquet
conn.execute("""
    COPY result TO 'data/customer_summary.parquet' (FORMAT PARQUET)
""")
```

---

## Schema Evolution: Handling Changes Over Time

Source systems change. New columns appear, old ones are renamed, types change. Handle this gracefully.

```python
def safe_merge_schemas(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle schema evolution: add new columns from new data, keep old columns.
    New columns get null values in old rows. Old columns missing from new data are kept.
    """
    new_cols = set(new_df.columns) - set(existing_df.columns)
    missing_cols = set(existing_df.columns) - set(new_df.columns)

    if new_cols:
        print(f"New columns detected: {new_cols}. Adding to schema.")
        for col in new_cols:
            existing_df[col] = None

    if missing_cols:
        print(f"Columns missing from new data: {missing_cols}. Keeping nulls.")
        for col in missing_cols:
            new_df[col] = None

    return pd.concat([existing_df, new_df[existing_df.columns]], ignore_index=True)
```

---

## Common Data Engineering Anti-Patterns

**Full refresh every run** — Loading all data from scratch daily is fine at small scale, but breaks at millions of rows. Design for incremental loads from day one.

**No data quality checks** — Pipelines that silently ingest corrupt data are worse than pipelines that fail loudly. Always validate at load time.

**Hardcoded credentials** — Never put passwords, API keys, or connection strings in code. Use environment variables, secrets managers (AWS Secrets Manager, HashiCorp Vault), or `.env` files that are gitignored.

**Not logging pipeline runs** — Without audit logs, debugging failures is guesswork. Log the run start time, end time, rows processed, and any errors to a `pipeline_runs` table.

**Schema coupling to source** — Building pipelines that fail when the source adds a column. Design transformations to be additive-compatible.

**Ignoring time zones** — Store all timestamps in UTC. Convert to local time only at the presentation layer. Mixing UTC and local time causes silent aggregation errors that are extremely hard to debug.
