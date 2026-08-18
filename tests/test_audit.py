import threading

from shared.audit import AuditStore


def test_audit_store_uses_wal_and_prunes_payload_quota(tmp_path) -> None:
    store = AuditStore(
        tmp_path / "audit.db",
        payload_dir=tmp_path / "audit_payloads",
    )
    payload_directory = store.allocate_payload_dir()
    (payload_directory / "passport.png").write_bytes(b"payload")
    relative = payload_directory.relative_to(store.payload_dir.parent)
    store.record(
        method="POST",
        path="/v1/passport",
        service="passport",
        status_code=200,
        latency_ms=1,
        payload_dir=str(relative),
    )

    with store._connect() as connection:
        journal_mode = connection.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0]
    result = store.prune(retention_days=30, max_payload_bytes=0)

    assert str(journal_mode).lower() == "wal"
    assert result["payload_directories"] == 1
    assert not payload_directory.exists()
    assert store.recent_events(hours=None)[0].payload_dir is None


def test_audit_store_handles_concurrent_sqlite_writes(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.db")

    def write_event(index: int) -> None:
        store.record(
            method="POST",
            path="/v1/passport",
            service="passport",
            status_code=200,
            latency_ms=float(index),
        )

    threads = [
        threading.Thread(target=write_event, args=(index,))
        for index in range(10)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.summary(hours=None)["total"] == 10
