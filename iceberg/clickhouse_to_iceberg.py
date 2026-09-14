"""Backfill alerts from ClickHouse (last 90 days) to Iceberg."""

import datetime
from collections.abc import Callable, Iterable

from iceberg.writer import AlertIcebergWriter

RETENTION_DAYS = 90


def backfill_window(days: int = RETENTION_DAYS) -> tuple[str, str]:
    """Return (start_date, end_date) ISO window for the backfill."""
    end = datetime.datetime.now(datetime.timezone.utc).date()
    start = end - datetime.timedelta(days=days)
    return (start.isoformat(), end.isoformat())


def run(
    writer: AlertIcebergWriter,
    fetch_alerts: Callable[[str, str], Iterable[dict]],
    days: int = RETENTION_DAYS,
) -> int:
    """Fetch window and append alerts; return archived count."""
    # ponytail: production fetch_alerts is a clickhouse-connect query on
    # transactions_raw/alerts; production writer wraps the pyiceberg table.
    start, end = backfill_window(days)
    count = 0
    for alert in fetch_alerts(start, end):
        try:
            writer.append(alert)
            count += 1
        except FileExistsError:
            continue
    return count
