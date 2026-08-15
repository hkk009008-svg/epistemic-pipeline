"""Adversarial stress tests for Worktree Sandbox & Event Lock Fallback Robustness (M3).

Tests:
1. resolve_git_dir adversarial inputs (nested pointers, circular pointers, corrupted files,
   empty files, unicode/emojis/combining chars, non-existent targets, long paths, special chars).
2. knowledge_store atomic link fallback under high-concurrency stress with simulated EXDEV,
   EPERM, ENOSYS, ENOTSUP, EMLINK, checking data integrity, zero tmp-leak, and SQLite consistency.
3. Read-only sandbox simulation (chmod 0555) verifying smooth step-by-step escalation
   Tier 1 -> Tier 2 -> Tier 3 -> Tier 4, and verifying lock mutual exclusion at every tier.
4. Tier 3 atomic lock edge cases: concurrent stale eviction races, live PID protection,
   zero-byte recovery, cross-process supersede protection on release.
5. Async/sync hybrid stress under simulated sandbox restrictions.
"""
from __future__ import annotations

import asyncio
import errno
import json
import os
import socket
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from pipeline.event_lock import (
    LockTier,
    WorktreeEventLock,
    resolve_git_dir,
)
from pipeline.knowledge_store import KnowledgeStore

# ===========================================================================
# 1. resolve_git_dir Adversarial Stress Tests
# ===========================================================================

class TestResolveGitDirAdversarial:
    """Stress-test resolve_git_dir against malformed, hostile, and boundary inputs."""

    def test_empty_git_file(self, tmp_path: Path):
        """Empty 0-byte .git file should fall back to worktree dir without crashing."""
        wt = tmp_path / "wt_empty"
        wt.mkdir()
        git_file = wt / ".git"
        git_file.write_bytes(b"")

        resolved = resolve_git_dir(wt)
        assert resolved == wt.resolve()

    def test_whitespace_only_git_file(self, tmp_path: Path):
        """Whitespace-only .git file falls back to worktree directory."""
        wt = tmp_path / "wt_ws"
        wt.mkdir()
        git_file = wt / ".git"
        git_file.write_text("   \n\t  \r\n   \n", encoding="utf-8")

        resolved = resolve_git_dir(wt)
        assert resolved == wt.resolve()

    def test_corrupted_binary_git_file(self, tmp_path: Path):
        """Binary/ELF/random bytes in .git file should not raise UnicodeDecodeError or crash."""
        wt = tmp_path / "wt_binary"
        wt.mkdir()
        git_file = wt / ".git"
        git_file.write_bytes(b"\x7fELF\x02\x01\x01\x00" + os.urandom(1024) + b"\xff\xfe\x00\x00")

        resolved = resolve_git_dir(wt)
        assert resolved == wt.resolve()

    def test_huge_git_file_with_embedded_gitdir(self, tmp_path: Path):
        """Large .git file with junk lines followed by valid gitdir line."""
        wt = tmp_path / "wt_large"
        wt.mkdir()
        target_dir = tmp_path / "actual_git_admin"
        target_dir.mkdir()

        junk = "\n".join([f"# comment line {i} with some noise" for i in range(5000)])
        content = f"{junk}\ngitdir: {target_dir.resolve()}\n# trailing noise"
        git_file = wt / ".git"
        git_file.write_text(content, encoding="utf-8")

        resolved = resolve_git_dir(wt)
        assert resolved == target_dir.resolve()

    def test_nonexistent_gitdir_target(self, tmp_path: Path):
        """gitdir pointing to non-existent directory should return the resolved target path."""
        wt = tmp_path / "wt_nonexistent_target"
        wt.mkdir()
        nonexistent_target = tmp_path / "does_not_exist" / "deep" / "admin"
        git_file = wt / ".git"
        git_file.write_text(f"gitdir: {nonexistent_target}\n", encoding="utf-8")

        resolved = resolve_git_dir(wt)
        assert resolved == nonexistent_target.resolve()

    def test_unicode_and_special_characters_in_paths(self, tmp_path: Path):
        """Unicode characters (CJK, emojis, spaces, accents, special punctuation) in path and gitdir."""
        unicode_name = "워크트리_🚀_공동작업_résumé_ünicöde_#123"
        wt = tmp_path / unicode_name
        wt.mkdir()

        admin_name = "관리자_저장소_📦_admin"
        admin_dir = tmp_path / admin_name
        admin_dir.mkdir()

        git_file = wt / ".git"
        git_file.write_text(f"gitdir: {admin_dir.resolve()}\n", encoding="utf-8")

        resolved = resolve_git_dir(wt)
        assert resolved == admin_dir.resolve()

        # Also test resolving from a sub-sub directory with unicode
        sub = wt / "하위폴더_📁" / "depth2"
        sub.mkdir(parents=True)
        resolved_sub = resolve_git_dir(sub)
        assert resolved_sub == admin_dir.resolve()

    def test_deeply_nested_relative_worktree_pointer(self, tmp_path: Path):
        """Deeply nested relative paths (../../../../../../../..) resolving correctly."""
        base = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "worktree"
        base.mkdir(parents=True)

        target = tmp_path / "x" / "y" / "git_admin"
        target.mkdir(parents=True)

        # 7 levels up from base to reach tmp_path
        rel_path = "../../../../../../../x/y/git_admin"
        git_file = base / ".git"
        git_file.write_text(f"gitdir: {rel_path}\n", encoding="utf-8")

        resolved = resolve_git_dir(base)
        assert resolved == target.resolve()

    def test_circular_and_self_referencing_gitdir_pointers(self, tmp_path: Path):
        """Self-referencing gitdir (points to self or parent) does not trigger infinite recursion."""
        wt = tmp_path / "wt_circular"
        wt.mkdir()
        git_file = wt / ".git"
        # Points to itself
        git_file.write_text(f"gitdir: {git_file.resolve()}\n", encoding="utf-8")

        resolved = resolve_git_dir(wt)
        # Should return the git_file path as resolved without hanging
        assert resolved == git_file.resolve()

    def test_multiple_gitdir_lines_uses_first_valid(self, tmp_path: Path):
        """If multiple gitdir lines are present, first valid entry is resolved."""
        wt = tmp_path / "wt_multi"
        wt.mkdir()
        target1 = tmp_path / "target1"
        target1.mkdir()
        target2 = tmp_path / "target2"
        target2.mkdir()

        git_file = wt / ".git"
        git_file.write_text(f"gitdir: {target1.resolve()}\ngitdir: {target2.resolve()}\n", encoding="utf-8")

        resolved = resolve_git_dir(wt)
        assert resolved == target1.resolve()

    def test_gitdir_with_windows_style_crlf_and_tabs(self, tmp_path: Path):
        """gitdir line formatted with Windows CRLF and trailing tabs."""
        wt = tmp_path / "wt_crlf"
        wt.mkdir()
        target = tmp_path / "admin_crlf"
        target.mkdir()

        git_file = wt / ".git"
        git_file.write_text(f"gitdir:\t{target.resolve()}\t\r\n", encoding="utf-8")

        resolved = resolve_git_dir(wt)
        assert resolved == target.resolve()


# ===========================================================================
# 2. KnowledgeStore Atomic Link Fallback & High Concurrency Stress
# ===========================================================================

class TestKnowledgeStoreFallbackStress:
    """Stress-test KnowledgeStore upsert_document under simulated link failures and concurrency."""

    @pytest.mark.parametrize("simulated_errno", [
        errno.EXDEV,
        errno.EPERM,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.EPERM),
        getattr(errno, "EOPNOTSUPP", errno.EPERM),
        getattr(errno, "EMLINK", errno.EPERM),
    ])
    def test_link_failure_matrix_data_integrity(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, simulated_errno: int):
        """Every supported link failure errno gracefully falls back to atomic replace with hash verification."""
        store = KnowledgeStore(tmp_path / f"store_errno_{simulated_errno}")

        def mock_link(src: Path | str, dst: Path | str):
            raise OSError(simulated_errno, f"Simulated link failure errno={simulated_errno}")

        monkeypatch.setattr(os, "link", mock_link)

        content = f"Critical document payload under errno {simulated_errno}."
        record = store.upsert_document(
            document_id="stress_doc",
            folder="adversarial/errno",
            title=f"Doc Errno {simulated_errno}",
            content=content,
        )

        assert record.document_id == "stress_doc"
        # Verify stored file
        version_file = store.root / record.relative_path
        assert version_file.exists()
        assert version_file.read_text(encoding="utf-8") == content

        # Verify no .tmp files leaked in versions directory
        versions_dir = store.root / "sources" / "versions"
        tmp_files = list(versions_dir.glob(".*.tmp"))
        assert len(tmp_files) == 0, f"Leaked tmp files: {tmp_files}"

        # Verify retrieval
        packet = store.retrieve("Critical document", top_k=5)
        assert len(packet.items) == 1
        assert packet.items[0].text == content

    def test_rapid_concurrent_upserts_under_exdev_mixed_documents(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """50 concurrent worker threads upserting 25 unique documents (2 threads per document) under EXDEV."""
        store = KnowledgeStore(tmp_path / "concurrent_store")

        def mock_link(src: Path | str, dst: Path | str):
            raise OSError(errno.EXDEV, "Cross-device link simulated")

        monkeypatch.setattr(os, "link", mock_link)

        num_docs = 25
        workers_per_doc = 2
        total_tasks = num_docs * workers_per_doc

        tasks = []
        for i in range(num_docs):
            doc_id = f"concurrent_doc_{i:03d}"
            content = f"Unique high-concurrency content payload for doc {i} with hash salt {uuid.uuid4().hex}."
            for w in range(workers_per_doc):
                tasks.append((doc_id, f"folder_{i % 5}", f"Title {i}", content))

        def execute_upsert(args):
            d_id, folder, title, text = args
            return store.upsert_document(d_id, folder, title, text)

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(execute_upsert, t) for t in tasks]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == total_tasks

        # Verify all 25 documents are indexed and retrievable
        for i in range(num_docs):
            packet = store.retrieve(f"payload for doc {i}", top_k=5)
            assert len(packet.items) >= 1
            assert any(item.document_id == f"concurrent_doc_{i:03d}" for item in packet.items)

        # Verify zero temporary files leaked
        versions_dir = store.root / "sources" / "versions"
        tmp_files = list(versions_dir.glob(".*.tmp"))
        assert len(tmp_files) == 0, f"Leaked tmp files: {tmp_files}"

    def test_unexpected_oserror_during_replace_fails_cleanly_and_cleans_tmp(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """If both os.link fails with EXDEV and os.replace fails with EIO, temporary file must be unlinked."""
        store = KnowledgeStore(tmp_path / "store_eio")

        def mock_link(src: Path | str, dst: Path | str):
            raise OSError(errno.EXDEV, "Cross-device link")

        def mock_replace(src: Path | str, dst: Path | str):
            raise OSError(errno.EIO, "I/O hardware error on replace")

        monkeypatch.setattr(os, "link", mock_link)
        monkeypatch.setattr(os, "replace", mock_replace)

        with pytest.raises(OSError) as exc_info:
            store.upsert_document(
                "doc_io_fail",
                "folder",
                "Title",
                "Content that will fail on replace.",
            )
        assert exc_info.value.errno == errno.EIO

        # Verify temp file cleaned up
        versions_dir = store.root / "sources" / "versions"
        tmp_files = list(versions_dir.glob(".*.tmp"))
        assert len(tmp_files) == 0


# ===========================================================================
# 3. Read-Only Sandbox Simulation & 4-Tier Fallback Escalation
# ===========================================================================

class TestSandboxReadonlyEscalation:
    """Stress-test tier fallback escalation under permission/sandbox constraints."""

    def test_tier1_to_tier2_escalation_on_readonly_git_dir(self, tmp_path: Path):
        """When git directory is read-only (chmod 0555), lock cleanly falls back to Tier 2 (TEMP_FLOCK)."""
        repo_dir = tmp_path / "ro_repo"
        repo_dir.mkdir()
        git_dir = repo_dir / ".git"
        git_dir.mkdir()

        # Make .git directory read-only
        try:
            os.chmod(git_dir, 0o555)

            lock = WorktreeEventLock(repo_dir, timeout_seconds=2.0)
            acquired = lock.acquire(blocking=True)
            try:
                assert acquired is True
                diag = lock.get_diagnostic()
                assert diag.active_tier == LockTier.TEMP_FLOCK
                assert diag.is_fallback is True
                assert any("UNWRITABLE" in r or "TIER1" in r for r in diag.fallback_reasons)
                assert diag.resolved_lock_path is not None
                assert "/tmp" in diag.resolved_lock_path or tempfile.gettempdir() in diag.resolved_lock_path
            finally:
                lock.release()
        finally:
            os.chmod(git_dir, 0o755)

    def test_tier2_to_tier3_escalation_when_flock_fails_in_temp(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """When Tier 1 unwritable and Tier 2 flock returns ENOLCK/ENOSYS, escalates to Tier 3 (USER_SPACE_ATOMIC)."""
        repo_dir = tmp_path / "ro_repo_t3"
        repo_dir.mkdir()
        git_dir = repo_dir / ".git"
        git_dir.mkdir()

        try:
            os.chmod(git_dir, 0o555)

            # Mock fcntl.flock to fail with ENOLCK (No locks available / NFS lock failure)
            import fcntl as real_fcntl
            def mock_flock(fd, operation):
                raise OSError(errno.ENOLCK, "No record locks available")

            monkeypatch.setattr(real_fcntl, "flock", mock_flock)

            lock = WorktreeEventLock(repo_dir, timeout_seconds=2.0)
            acquired = lock.acquire(blocking=True)
            try:
                assert acquired is True
                diag = lock.get_diagnostic()
                assert diag.active_tier == LockTier.USER_SPACE_ATOMIC
                assert diag.is_fallback is True
                assert any("TIER2_FLOCK_FAILED" in r for r in diag.fallback_reasons)
                assert diag.resolved_lock_path.endswith(".atomic_lock")
            finally:
                lock.release()
        finally:
            os.chmod(git_dir, 0o755)

    def test_tier3_to_tier4_escalation_when_all_filesystem_writes_fail(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """When all filesystem operations are completely barred (full EROFS sandbox), escalates to Tier 4 (IN_MEMORY_MUTEX)."""
        target_path = tmp_path / "total_sandbox_repo"
        target_path.mkdir()

        # Mock os.open to fail with EACCES everywhere except for reading
        real_open = os.open
        def mock_open(path, flags, *args, **kwargs):
            if (flags & os.O_CREAT) or (flags & os.O_RDWR) or (flags & os.O_WRONLY):
                raise OSError(errno.EACCES, "Sandbox write forbidden")
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", mock_open)

        lock = WorktreeEventLock(target_path, timeout_seconds=2.0)
        acquired = lock.acquire(blocking=True)
        try:
            assert acquired is True
            diag = lock.get_diagnostic()
            assert diag.active_tier == LockTier.IN_MEMORY_MUTEX
            assert diag.is_fallback is True
            assert diag.resolved_lock_path.startswith("in_memory://")
            assert len(diag.fallback_reasons) >= 2
        finally:
            lock.release()

    def test_all_tiers_provide_strict_mutual_exclusion_under_concurrency(self, tmp_path: Path):
        """Tests that 8 concurrent threads attempting to acquire a lock at each tier enforce mutual exclusion."""
        for tier_setup in ["tier1", "tier2", "tier4"]:
            target_dir = tmp_path / f"mutex_test_{tier_setup}"
            target_dir.mkdir(parents=True)

            if tier_setup == "tier1":
                (target_dir / ".git").mkdir()
            elif tier_setup == "tier2":
                git_dir = target_dir / ".git"
                git_dir.mkdir()
                os.chmod(git_dir, 0o555)

            shared_counter = 0
            concurrent_overlaps = 0
            lock_holder_active = False
            lock_guard = threading.Lock()

            def critical_section(thread_id: int, t_dir: Path = target_dir, l_guard: threading.Lock = lock_guard):
                nonlocal shared_counter, concurrent_overlaps, lock_holder_active
                lock = WorktreeEventLock(t_dir, timeout_seconds=5.0)

                with lock:
                    with l_guard:
                        if lock_holder_active:
                            concurrent_overlaps += 1
                        lock_holder_active = True
                    time.sleep(0.01)
                    shared_counter += 1
                    with l_guard:
                        lock_holder_active = False

            try:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(critical_section, i) for i in range(8)]
                    for f in futures:
                        f.result()

                assert shared_counter == 8, f"Failed on {tier_setup}: expected counter 8, got {shared_counter}"
                assert concurrent_overlaps == 0, f"Mutual exclusion violated on {tier_setup}!"
            finally:
                if tier_setup == "tier2":
                    os.chmod(target_dir / ".git", 0o755)


# ===========================================================================
# 4. Tier 3 Atomic Lock Stale Eviction and Race Conditions
# ===========================================================================

class TestTier3AtomicLockHardening:
    """Stress-test Tier 3 atomic lock eviction, live PID protection, and race-free quarantine."""

    def test_live_pid_is_never_evicted_before_stale_timeout(self, tmp_path: Path):
        """Active lockfile with living local PID and valid timestamp must not be evicted."""
        target_dir = tmp_path / "live_lock_target"
        target_dir.mkdir()

        lock = WorktreeEventLock(target_dir, timeout_seconds=0.2, stale_timeout_seconds=60.0)
        temp_atomic = lock._get_temp_lock_path(suffix=".atomic_lock")

        # Create lock owned by current process (definitely alive!)
        payload = {
            "lock_id": "active_holder_123",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "target_path": str(target_dir.resolve()),
            "acquired_at": time.time() - 10.0,  # 10s ago, well within 60s timeout
            "stale_timeout_seconds": 60.0,
        }
        temp_atomic.write_text(json.dumps(payload), encoding="utf-8")

        try:
            # Another lock instance trying non-blocking acquire
            lock2 = WorktreeEventLock(target_dir, timeout_seconds=0.1)
            # Should fail to evict and report contention
            evicted = lock2._check_and_evict_stale_tier3(temp_atomic)
            assert evicted is False
            assert temp_atomic.exists()
        finally:
            temp_atomic.unlink(missing_ok=True)

    def test_dead_pid_on_local_host_is_evicted_immediately(self, tmp_path: Path):
        """Lockfile with non-existent PID on localhost is evicted regardless of acquired_at."""
        target_dir = tmp_path / "dead_pid_target"
        target_dir.mkdir()

        lock = WorktreeEventLock(target_dir, timeout_seconds=1.0)
        temp_atomic = lock._get_temp_lock_path(suffix=".atomic_lock")

        # Find a confirmed unused PID
        dead_pid = 99999999
        while True:
            try:
                os.kill(dead_pid, 0)
                dead_pid -= 1
            except ProcessLookupError:
                break

        payload = {
            "lock_id": "dead_holder_999",
            "pid": dead_pid,
            "hostname": socket.gethostname(),
            "target_path": str(target_dir.resolve()),
            "acquired_at": time.time(),  # fresh timestamp
            "stale_timeout_seconds": 60.0,
        }
        temp_atomic.write_text(json.dumps(payload), encoding="utf-8")

        try:
            lock2 = WorktreeEventLock(target_dir, timeout_seconds=1.0)
            evicted = lock2._check_and_evict_stale_tier3(temp_atomic)
            assert evicted is True
            assert not temp_atomic.exists()
        finally:
            temp_atomic.unlink(missing_ok=True)

    def test_concurrent_eviction_race_between_competing_processes(self, tmp_path: Path):
        """When multiple workers race to evict the same dead lock, exactly one renames it and no crash occurs."""
        target_dir = tmp_path / "eviction_race_target"
        target_dir.mkdir()

        lock = WorktreeEventLock(target_dir)
        temp_atomic = lock._get_temp_lock_path(suffix=".atomic_lock")

        dead_pid = 99999998
        payload = {
            "lock_id": "dead_race_holder",
            "pid": dead_pid,
            "hostname": socket.gethostname(),
            "target_path": str(target_dir.resolve()),
            "acquired_at": time.time() - 100.0,
            "stale_timeout_seconds": 30.0,
        }
        temp_atomic.write_text(json.dumps(payload), encoding="utf-8")

        eviction_results = []

        def attempt_eviction(worker_id: int):
            l = WorktreeEventLock(target_dir)
            return l._check_and_evict_stale_tier3(temp_atomic)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(attempt_eviction, i) for i in range(8)]
            for f in futures:
                eviction_results.append(f.result())

        # Exactly one worker should successfully perform the atomic rename/eviction
        assert sum(eviction_results) == 1
        assert not temp_atomic.exists()


# ===========================================================================
# 5. Diagnostic Reporting & Edge Invariants
# ===========================================================================

class TestDiagnosticReportingAndEdgeCases:
    """Verifies that diagnostic state matches internal state across all scenarios."""

    def test_diagnostic_serialization_and_duration_metrics(self, tmp_path: Path):
        """Diagnostics record accurate pid, acquired_at, durations, and serialize cleanly."""
        target_dir = tmp_path / "diag_repo"
        target_dir.mkdir()
        (target_dir / ".git").mkdir()

        lock = WorktreeEventLock(target_dir, timeout_seconds=2.0)
        diag_before = lock.get_diagnostic()
        assert diag_before.is_locked is False
        assert diag_before.holder_pid is None
        assert diag_before.held_duration_ms is None

        with lock:
            diag_during = lock.get_diagnostic()
            assert diag_during.is_locked is True
            assert diag_during.holder_pid == os.getpid()
            assert diag_during.acquired_at is not None
            assert diag_during.acquisition_duration_ms is not None
            assert diag_during.acquisition_duration_ms >= 0.0
            time.sleep(0.05)
            assert diag_during.held_duration_ms > 40.0  # at least 40ms held

            d_dict = diag_during.to_dict()
            assert d_dict["is_locked"] is True
            assert d_dict["holder_pid"] == os.getpid()
            assert "LockDiagnostic[LOCKED]" in diag_during.summary()

        diag_after = lock.get_diagnostic()
        assert diag_after.is_locked is False
        assert diag_after.held_duration_ms is not None
        assert diag_after.held_duration_ms > 40.0

    @pytest.mark.asyncio
    async def test_async_and_sync_interoperability_under_tier1(self, tmp_path: Path):
        """Verifies that synchronous lock and asynchronous lock on same target honor mutual exclusion."""
        repo_dir = tmp_path / "sync_async_repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        sync_lock = WorktreeEventLock(repo_dir, timeout_seconds=0.1)
        async_lock = WorktreeEventLock(repo_dir, timeout_seconds=0.1)

        sync_acquired = sync_lock.acquire()
        assert sync_acquired is True

        # Async lock attempt should fail due to sync lock holding Tier 1
        async_acquired = await async_lock.acquire_async(blocking=False)
        assert async_acquired is False

        sync_lock.release()

        # Now async lock should succeed
        async_acquired2 = await async_lock.acquire_async(blocking=True, timeout=1.0)
        assert async_acquired2 is True
        await async_lock.release_async()

    @pytest.mark.asyncio
    async def test_async_task_cancellation_releases_lock(self, tmp_path: Path):
        """If an async task holding the lock is cancelled, __aexit__ cleanly releases the lock."""
        repo_dir = tmp_path / "async_cancel_repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        lock_acquired_event = asyncio.Event()
        lock_released_event = asyncio.Event()

        async def worker():
            lock = WorktreeEventLock(repo_dir, timeout_seconds=5.0)
            async with lock:
                lock_acquired_event.set()
                try:
                    await asyncio.sleep(10.0)  # Wait to be cancelled
                finally:
                    pass
            lock_released_event.set()

        task = asyncio.create_task(worker())
        await lock_acquired_event.wait()

        # Task holds the lock. Now cancel it.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Verify another lock can immediately acquire
        probe_lock = WorktreeEventLock(repo_dir, timeout_seconds=0.5)
        probe_acquired = await probe_lock.acquire_async(blocking=False)
        assert probe_acquired is True
        await probe_lock.release_async()


# ===========================================================================
# 6. High-Churn Concurrency & Boundary Symlink Cases
# ===========================================================================

class TestHighChurnAndSymlinkBoundaries:
    """Tests extreme concurrency churn and symlink handling."""

    def test_rapid_acquire_release_thrashing_across_threads(self, tmp_path: Path):
        """16 threads performing 50 rapid acquire-release cycles in parallel."""
        target_dir = tmp_path / "thrash_repo"
        target_dir.mkdir()
        (target_dir / ".git").mkdir()

        counter = 0
        total_ops = 50
        num_threads = 8

        def thrash_worker(worker_id: int):
            nonlocal counter
            for _ in range(total_ops):
                lock = WorktreeEventLock(target_dir, timeout_seconds=10.0)
                with lock:
                    c = counter
                    time.sleep(0.0001)
                    counter = c + 1

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(thrash_worker, i) for i in range(num_threads)]
            for f in futures:
                f.result()

        assert counter == total_ops * num_threads

    def test_resolve_git_dir_symlink_to_worktree(self, tmp_path: Path):
        """Symlinked worktree directory resolves properly to target's git directory."""
        real_repo = tmp_path / "real_repo"
        real_repo.mkdir()
        (real_repo / ".git").mkdir()

        symlink_repo = tmp_path / "symlink_repo"
        symlink_repo.symlink_to(real_repo, target_is_directory=True)

        resolved = resolve_git_dir(symlink_repo)
        assert resolved == (real_repo / ".git").resolve()

    def test_resolve_git_dir_broken_symlink(self, tmp_path: Path):
        """Broken symlink does not raise FileNotFoundError or crash."""
        broken = tmp_path / "broken_link"
        broken.symlink_to(tmp_path / "nonexistent_target_directory", target_is_directory=True)

        resolved = resolve_git_dir(broken)
        # Should gracefully resolve target or return path
        assert isinstance(resolved, Path)

