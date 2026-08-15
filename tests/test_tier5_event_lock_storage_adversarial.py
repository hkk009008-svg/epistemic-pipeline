"""Tier 5 Adversarial Stress & Empirical Hardening Test Suite.

Covers:
1. Worktree Event Lock:
   - Multi-process heavy contention (10-16 processes) across Tier 1, Tier 2, Tier 3.
   - Abrupt SIGKILL crash simulations during lock hold for Tier 1, Tier 2, Tier 3.
   - Race-free atomic quarantine rename protocol for dead PIDs and concurrent recovery.
   - Stale lock TTL eviction under heavy contention (live PID timeout, remote host timeout).
   - Corrupted and unreadable lockfile recovery (0-byte, truncated JSON, binary noise, fresh vs stale).
   - Full 4-tier fallback hierarchy progression (KERNEL_FLOCK -> TEMP_FLOCK -> USER_SPACE_ATOMIC -> IN_MEMORY_MUTEX).
   - Read-only filesystem simulations (EROFS, chmod 0555).
   - Deep nested worktree .git pointer resolution, circular loops, and ascending traversal.
2. KnowledgeStore Storage Resilience:
   - Atomic publication fallback under full errno matrix (EXDEV, EPERM, ENOSYS, ENOTSUP, EOPNOTSUPP, EMLINK).
   - Double-failure clean rollback and temporary file leak prevention.
   - Byte-level SHA-256 integrity verification and post-write tampering detection.
   - High-concurrency multi-process & multi-thread publications, revision conflict detection,
     idempotent deduplication, and mixed WAL read/write throughput.
"""
from __future__ import annotations

import errno
import json
import multiprocessing as mp
import os
import signal
import socket
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pipeline.event_lock import (
    LockTier,
    WorktreeEventLock,
    resolve_git_dir,
)
from pipeline.knowledge_store import (
    DocumentRecord,
    DocumentVersionConflictError,
    KnowledgeStore,
    KnowledgeStoreError,
)

# ===========================================================================
# Top-Level Multiprocessing Helpers (macOS spawn safe)
# ===========================================================================

def _mp_counter_worker(
    worktree_path: str,
    counter_file: str,
    iterations: int,
    force_tier: str | None,
    result_queue: mp.Queue,
    hold_delay: float = 0.001,
) -> None:
    """Worker process: acquires lock, reads counter, increments, writes back."""
    pid = os.getpid()
    tiers_observed: set[str] = set()

    if force_tier == "tier2":
        # Handled by caller making .git unwritable
        pass
    elif force_tier == "tier3":
        import pipeline.event_lock as el
        el.fcntl = None  # Disable fcntl in this child process

    for _ in range(iterations):
        lock = WorktreeEventLock(
            worktree_path,
            timeout_seconds=30.0,
            poll_interval=0.005,
        )
        if not lock.acquire(blocking=True):
            result_queue.put((pid, False, f"Timeout in PID {pid}", list(tiers_observed)))
            return

        try:
            tiers_observed.add(lock.active_tier.value if lock.active_tier else "unknown")
            val = 0
            cpath = Path(counter_file)
            if cpath.exists():
                content = cpath.read_text(encoding="utf-8").strip()
                if content:
                    val = int(content)

            if hold_delay > 0:
                time.sleep(hold_delay)

            cpath.write_text(str(val + 1), encoding="utf-8")
        finally:
            lock.release()

    result_queue.put((pid, True, iterations, list(tiers_observed)))


def _mp_hold_lock_and_signal(
    worktree_path: str,
    force_tier: str | None,
    ready_file: str,
) -> None:
    """Acquires lock, writes PID to ready_file, and sleeps until SIGKILL."""
    if force_tier == "tier3":
        import pipeline.event_lock as el
        el.fcntl = None

    lock = WorktreeEventLock(worktree_path, timeout_seconds=15.0)
    if lock.acquire(blocking=True):
        Path(ready_file).write_text(str(os.getpid()), encoding="utf-8")
        while True:
            time.sleep(0.5)


def _mp_quarantine_race_worker(
    worktree_path: str,
    counter_file: str,
    result_queue: mp.Queue,
) -> None:
    """Competes for lock where dead PID lockfile exists in Tier 3."""
    import pipeline.event_lock as el
    el.fcntl = None

    pid = os.getpid()
    lock = WorktreeEventLock(worktree_path, timeout_seconds=10.0, poll_interval=0.005)
    acquired = lock.acquire(blocking=True)
    if not acquired:
        result_queue.put((pid, False, "Acquisition timeout"))
        return

    try:
        val = 0
        cpath = Path(counter_file)
        if cpath.exists():
            content = cpath.read_text(encoding="utf-8").strip()
            if content:
                val = int(content)
        time.sleep(0.002)
        cpath.write_text(str(val + 1), encoding="utf-8")
        result_queue.put((pid, True, "OK"))
    finally:
        lock.release()


def _mp_ks_upsert_worker(
    store_root: str,
    doc_id: str,
    folder: str,
    title: str,
    content: str,
    simulate_exdev: bool,
    result_queue: mp.Queue,
) -> None:
    """Worker process: upserts a document into KnowledgeStore."""
    if simulate_exdev:
        def mock_link(src: Path | str, dst: Path | str):
            raise OSError(errno.EXDEV, "Simulated cross-device link")
        os.link = mock_link  # type: ignore

    store = KnowledgeStore(store_root)
    try:
        record = store.upsert_document(
            document_id=doc_id,
            folder=folder,
            title=title,
            content=content,
        )
        result_queue.put((os.getpid(), True, record.revision_id))
    except (OSError, RuntimeError, ValueError, KnowledgeStoreError) as exc:
        result_queue.put((os.getpid(), False, str(exc)))


# ===========================================================================
# 1. Worktree Event Lock Multi-Process Concurrency
# ===========================================================================

class TestEventLockMultiProcessConcurrency:
    """Stress tests verifying mutual exclusion and zero loss across multi-process workloads."""

    def test_tier1_multiprocess_high_contention(self, tmp_path: Path):
        """12 OS processes competing simultaneously for Tier 1 kernel flock."""
        repo = tmp_path / "mp_repo_tier1"
        (repo / ".git").mkdir(parents=True)
        counter_file = tmp_path / "counter_t1.txt"
        counter_file.write_text("0", encoding="utf-8")

        num_processes = 12
        iterations_per_process = 10
        expected_total = num_processes * iterations_per_process

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_mp_counter_worker,
                args=(str(repo), str(counter_file), iterations_per_process, None, queue, 0.001),
            )
            for _ in range(num_processes)
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join(timeout=35.0)
            assert not p.is_alive(), "Process hung during Tier 1 contention"

        results = [queue.get() for _ in range(num_processes)]
        for pid, ok, res, tiers in results:
            assert ok is True, f"Process {pid} failed: {res}"
            assert res == iterations_per_process
            assert LockTier.KERNEL_FLOCK.value in tiers

        final_count = int(counter_file.read_text(encoding="utf-8").strip())
        assert final_count == expected_total

    def test_tier2_multiprocess_high_contention_unwritable_git(self, tmp_path: Path):
        """12 OS processes competing when .git is unwritable, all falling back to Tier 2."""
        repo = tmp_path / "mp_repo_tier2"
        repo.mkdir()
        git_dir = repo / ".git"
        git_dir.mkdir()
        os.chmod(git_dir, 0o555)

        counter_file = tmp_path / "counter_t2.txt"
        counter_file.write_text("0", encoding="utf-8")

        num_processes = 12
        iterations_per_process = 8
        expected_total = num_processes * iterations_per_process

        try:
            ctx = mp.get_context("spawn")
            queue = ctx.Queue()
            processes = [
                ctx.Process(
                    target=_mp_counter_worker,
                    args=(str(repo), str(counter_file), iterations_per_process, "tier2", queue, 0.001),
                )
                for _ in range(num_processes)
            ]

            for p in processes:
                p.start()

            for p in processes:
                p.join(timeout=35.0)
                assert not p.is_alive(), "Process hung during Tier 2 contention"

            results = [queue.get() for _ in range(num_processes)]
            for pid, ok, res, tiers in results:
                assert ok is True, f"Process {pid} failed: {res}"
                assert res == iterations_per_process
                assert LockTier.TEMP_FLOCK.value in tiers

            final_count = int(counter_file.read_text(encoding="utf-8").strip())
            assert final_count == expected_total
        finally:
            os.chmod(git_dir, 0o755)

    def test_tier3_multiprocess_high_contention_atomic_lock(self, tmp_path: Path):
        """12 OS processes competing under Tier 3 atomic lockfile with fcntl disabled."""
        repo = tmp_path / "mp_repo_tier3"
        repo.mkdir()
        counter_file = tmp_path / "counter_t3.txt"
        counter_file.write_text("0", encoding="utf-8")

        num_processes = 12
        iterations_per_process = 8
        expected_total = num_processes * iterations_per_process

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_mp_counter_worker,
                args=(str(repo), str(counter_file), iterations_per_process, "tier3", queue, 0.001),
            )
            for _ in range(num_processes)
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join(timeout=35.0)
            assert not p.is_alive(), "Process hung during Tier 3 contention"

        results = [queue.get() for _ in range(num_processes)]
        for pid, ok, res, tiers in results:
            assert ok is True, f"Process {pid} failed: {res}"
            assert res == iterations_per_process
            assert LockTier.USER_SPACE_ATOMIC.value in tiers

        final_count = int(counter_file.read_text(encoding="utf-8").strip())
        assert final_count == expected_total

    def test_tier1_tier_isolation_under_extreme_contention(self, tmp_path: Path):
        """High-contention contention does NOT cause premature tier fallback."""
        repo = tmp_path / "isolation_repo"
        (repo / ".git").mkdir(parents=True)

        tiers_seen: list[LockTier] = []
        lock_guard = threading.Lock()

        def hammer():
            for _ in range(15):
                lock = WorktreeEventLock(repo, timeout_seconds=10.0, poll_interval=0.002)
                with lock:
                    with lock_guard:
                        tiers_seen.append(lock.active_tier)
                    time.sleep(0.001)

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(hammer) for _ in range(16)]
            for f in futures:
                f.result()

        assert len(tiers_seen) == 16 * 15
        assert all(t == LockTier.KERNEL_FLOCK for t in tiers_seen)


# ===========================================================================
# 2. Crash Simulations (SIGKILL) & Atomic Quarantine Eviction
# ===========================================================================

class TestEventLockCrashAndSigkillRecovery:
    """Stress tests verifying instant recovery and atomic quarantine after SIGKILL."""

    def test_tier1_sigkill_lock_holder_immediate_recovery(self, tmp_path: Path):
        """Process holding Tier 1 kernel flock killed with SIGKILL -> lock immediately released by OS."""
        repo = tmp_path / "sigkill_t1_repo"
        (repo / ".git").mkdir(parents=True)
        ready_file = tmp_path / "ready_t1.txt"

        ctx = mp.get_context("spawn")
        p = ctx.Process(
            target=_mp_hold_lock_and_signal,
            args=(str(repo), None, str(ready_file)),
        )
        p.start()

        try:
            # Wait for process to hold lock
            for _ in range(100):
                if ready_file.exists() and ready_file.read_text().strip():
                    break
                time.sleep(0.05)
            assert ready_file.exists()

            # Confirm lock is currently held
            probe = WorktreeEventLock(repo, timeout_seconds=0.1)
            assert probe.acquire(blocking=False) is False

            # Kill holder abruptly with SIGKILL
            os.kill(p.pid, signal.SIGKILL)
            p.join(timeout=5.0)

            # Next lock should acquire almost immediately (<0.5s)
            start_t = time.time()
            new_lock = WorktreeEventLock(repo, timeout_seconds=2.0)
            assert new_lock.acquire(blocking=True) is True
            assert (time.time() - start_t) < 1.0
            new_lock.release()
        finally:
            if p.is_alive():
                os.kill(p.pid, signal.SIGKILL)

    def test_tier2_sigkill_temp_flock_immediate_recovery(self, tmp_path: Path):
        """Process holding Tier 2 temp flock killed with SIGKILL -> lock immediately freed."""
        repo = tmp_path / "sigkill_t2_repo"
        repo.mkdir()
        git_dir = repo / ".git"
        git_dir.mkdir()
        os.chmod(git_dir, 0o555)
        ready_file = tmp_path / "ready_t2.txt"

        ctx = mp.get_context("spawn")
        p = ctx.Process(
            target=_mp_hold_lock_and_signal,
            args=(str(repo), "tier2", str(ready_file)),
        )
        p.start()

        try:
            for _ in range(100):
                if ready_file.exists() and ready_file.read_text().strip():
                    break
                time.sleep(0.05)
            assert ready_file.exists()

            # Kill holder abruptly
            os.kill(p.pid, signal.SIGKILL)
            p.join(timeout=5.0)

            # Acquire in Tier 2
            new_lock = WorktreeEventLock(repo, timeout_seconds=2.0)
            assert new_lock.acquire(blocking=True) is True
            assert new_lock.active_tier == LockTier.TEMP_FLOCK
            new_lock.release()
        finally:
            os.chmod(git_dir, 0o755)
            if p.is_alive():
                os.kill(p.pid, signal.SIGKILL)

    def test_tier3_sigkill_dead_pid_atomic_quarantine_eviction(self, tmp_path: Path):
        """Process holding Tier 3 atomic lock killed with SIGKILL -> dead PID evicted via quarantine rename."""
        repo = tmp_path / "sigkill_t3_repo"
        repo.mkdir()
        ready_file = tmp_path / "ready_t3.txt"

        ctx = mp.get_context("spawn")
        p = ctx.Process(
            target=_mp_hold_lock_and_signal,
            args=(str(repo), "tier3", str(ready_file)),
        )
        p.start()

        try:
            for _ in range(100):
                if ready_file.exists() and ready_file.read_text().strip():
                    break
                time.sleep(0.05)
            assert ready_file.exists()
            dead_pid = int(ready_file.read_text().strip())

            # Kill holder abruptly with SIGKILL
            os.kill(p.pid, signal.SIGKILL)
            p.join(timeout=5.0)

            # Confirm dead PID
            assert not WorktreeEventLock._is_pid_alive(dead_pid)

            # Lockfile exists on disk with dead PID
            probe_lock = WorktreeEventLock(repo)
            temp_atomic = probe_lock._get_temp_lock_path(suffix=".atomic_lock")
            assert temp_atomic.exists()

            # Now acquire with fcntl disabled to test Tier 3 quarantine eviction
            import pipeline.event_lock as el
            old_fcntl = el.fcntl
            el.fcntl = None
            try:
                new_lock = WorktreeEventLock(repo, timeout_seconds=5.0)
                acquired = new_lock.acquire(blocking=True)
                assert acquired is True
                assert new_lock.active_tier == LockTier.USER_SPACE_ATOMIC
                diag = new_lock.get_diagnostic()
                assert any("Evicted stale Tier 3 lock" in r for r in diag.fallback_reasons)
                new_lock.release()
            finally:
                el.fcntl = old_fcntl
        finally:
            if p.is_alive():
                os.kill(p.pid, signal.SIGKILL)

    def test_concurrent_dead_pid_quarantine_race(self, tmp_path: Path):
        """10 OS processes simultaneously discover a dead PID lockfile and race to evict and acquire."""
        repo = tmp_path / "dead_race_repo"
        repo.mkdir()
        counter_file = tmp_path / "dead_race_counter.txt"
        counter_file.write_text("0", encoding="utf-8")

        # Create dead-PID atomic lockfile
        dead_pid = 99999990
        while WorktreeEventLock._is_pid_alive(dead_pid):
            dead_pid -= 1

        probe = WorktreeEventLock(repo)
        temp_atomic = probe._get_temp_lock_path(suffix=".atomic_lock")
        payload = {
            "lock_id": "dead_race_starter",
            "pid": dead_pid,
            "hostname": socket.gethostname(),
            "target_path": str(repo.resolve()),
            "acquired_at": time.time(),
            "stale_timeout_seconds": 60.0,
        }
        temp_atomic.write_text(json.dumps(payload), encoding="utf-8")

        num_processes = 10
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_mp_quarantine_race_worker,
                args=(str(repo), str(counter_file), queue),
            )
            for _ in range(num_processes)
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join(timeout=30.0)
            assert not p.is_alive()

        results = [queue.get() for _ in range(num_processes)]
        for pid, ok, msg in results:
            assert ok is True, f"Process {pid} failed: {msg}"

        final_count = int(counter_file.read_text(encoding="utf-8").strip())
        assert final_count == num_processes


# ===========================================================================
# 3. Stale Lock TTL Eviction & Corrupted Lock Recovery
# ===========================================================================

class TestEventLockStaleTTLEvictionAndCorruptionRecovery:
    """Stress tests for TTL timeout eviction (live PID, remote host) and corrupted lock recovery."""

    def test_tier3_stale_ttl_eviction_with_live_pid_exceeded_timeout(self, tmp_path: Path):
        """Lock held by current live PID is evicted once elapsed time > stale_timeout_seconds."""
        repo = tmp_path / "stale_live_repo"
        repo.mkdir()

        lock = WorktreeEventLock(repo, stale_timeout_seconds=1.5)
        temp_atomic = lock._get_temp_lock_path(suffix=".atomic_lock")

        payload = {
            "lock_id": "live_pid_stale_123",
            "pid": os.getpid(),  # Definitely alive
            "hostname": socket.gethostname(),
            "target_path": str(repo.resolve()),
            "acquired_at": time.time() - 10.0,  # 10s ago > 1.5s timeout
            "stale_timeout_seconds": 1.5,
        }
        temp_atomic.write_text(json.dumps(payload), encoding="utf-8")

        try:
            evicted = lock._check_and_evict_stale_tier3(temp_atomic)
            assert evicted is True
            assert not temp_atomic.exists()
        finally:
            temp_atomic.unlink(missing_ok=True)

    def test_tier3_stale_ttl_eviction_remote_hostname(self, tmp_path: Path):
        """Lock held on another hostname with elapsed time > timeout is evicted."""
        repo = tmp_path / "stale_remote_repo"
        repo.mkdir()

        lock = WorktreeEventLock(repo, stale_timeout_seconds=2.0)
        temp_atomic = lock._get_temp_lock_path(suffix=".atomic_lock")

        payload = {
            "lock_id": "remote_holder_456",
            "pid": 12345,
            "hostname": "other-compute-node-99",
            "target_path": str(repo.resolve()),
            "acquired_at": time.time() - 30.0,
            "stale_timeout_seconds": 2.0,
        }
        temp_atomic.write_text(json.dumps(payload), encoding="utf-8")

        try:
            evicted = lock._check_and_evict_stale_tier3(temp_atomic)
            assert evicted is True
            assert not temp_atomic.exists()
        finally:
            temp_atomic.unlink(missing_ok=True)

    def test_tier3_corrupted_empty_lockfile_aged_vs_fresh(self, tmp_path: Path):
        """0-byte lockfile: fresh (<5s) is treated as active contention; aged (>5s) is evicted."""
        repo = tmp_path / "empty_lock_repo"
        repo.mkdir()

        lock = WorktreeEventLock(repo)
        temp_atomic = lock._get_temp_lock_path(suffix=".atomic_lock")

        # 1. Fresh empty file (mtime = now)
        temp_atomic.write_bytes(b"")
        try:
            evicted_fresh = lock._check_and_evict_stale_tier3(temp_atomic)
            assert evicted_fresh is False
            assert temp_atomic.exists()

            # 2. Aged empty file (mtime = 10s ago)
            past_time = time.time() - 10.0
            os.utime(temp_atomic, (past_time, past_time))

            evicted_aged = lock._check_and_evict_stale_tier3(temp_atomic)
            assert evicted_aged is True
            assert not temp_atomic.exists()
        finally:
            temp_atomic.unlink(missing_ok=True)

    def test_tier3_corrupted_truncated_json_and_binary_garbage(self, tmp_path: Path):
        """Garbage JSON / binary noise in lockfile with aged mtime is safely quarantined and evicted."""
        repo = tmp_path / "corrupt_lock_repo"
        repo.mkdir()

        lock = WorktreeEventLock(repo)
        temp_atomic = lock._get_temp_lock_path(suffix=".atomic_lock")

        corrupt_payloads = [
            b'{"lock_id": "incomplete_json_stream',
            b"\x80\xff\xfe\x00\x01\x88\x99\xaa\xbb\xcc\xdd\xee\xff",
            b'{"pid": 1234, "acquired_at": 1000.0}',  # aged timestamp
            b"   \r\n\t  \n  ",
        ]

        for payload in corrupt_payloads:
            temp_atomic.write_bytes(payload)
            past_time = time.time() - 12.0
            os.utime(temp_atomic, (past_time, past_time))

            evicted = lock._check_and_evict_stale_tier3(temp_atomic)
            assert evicted is True, f"Failed to evict corrupt payload: {payload[:20]!r}"
            assert not temp_atomic.exists()


# ===========================================================================
# 4. Multi-Tier Fallback Progression & Read-Only Sandbox Simulations
# ===========================================================================

class TestEventLockMultiTierProgressionAndReadOnly:
    """Stress tests for multi-tier fallback progression and read-only sandboxes."""

    def test_full_four_tier_fallback_step_by_step(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Steps through Tier 1 -> Tier 2 -> Tier 3 -> Tier 4 with complete diagnostic verification."""
        repo = tmp_path / "step_repo"
        repo.mkdir()
        git_dir = repo / ".git"
        git_dir.mkdir()

        # Step 1: Standard git repo -> Tier 1
        lock1 = WorktreeEventLock(repo, timeout_seconds=2.0)
        with lock1:
            diag1 = lock1.get_diagnostic()
            assert diag1.active_tier == LockTier.KERNEL_FLOCK
            assert diag1.is_fallback is False
            assert diag1.resolved_lock_path == str(git_dir / "worktree_event.lock")

        # Step 2: Unlink lockfile and make .git unwritable -> Tier 2
        (git_dir / "worktree_event.lock").unlink(missing_ok=True)
        os.chmod(git_dir, 0o555)
        try:
            lock2 = WorktreeEventLock(repo, timeout_seconds=2.0)
            with lock2:
                diag2 = lock2.get_diagnostic()
                assert diag2.active_tier == LockTier.TEMP_FLOCK
                assert diag2.is_fallback is True
                assert any("TIER1_PATH_ERROR" in r or "GIT_DIR_UNWRITABLE" in r for r in diag2.fallback_reasons)
                assert diag2.resolved_lock_path.startswith(tempfile.gettempdir())
        finally:
            os.chmod(git_dir, 0o755)

        # Step 3: Tier 1 unwritable + Tier 2 flock raises ENOLCK -> Tier 3
        (git_dir / "worktree_event.lock").unlink(missing_ok=True)
        os.chmod(git_dir, 0o555)
        try:
            import fcntl as real_fcntl
            def mock_flock_nolck(fd, op):
                raise OSError(errno.ENOLCK, "Simulated ENOLCK")
            monkeypatch.setattr(real_fcntl, "flock", mock_flock_nolck)

            lock3 = WorktreeEventLock(repo, timeout_seconds=2.0)
            with lock3:
                diag3 = lock3.get_diagnostic()
                assert diag3.active_tier == LockTier.USER_SPACE_ATOMIC
                assert diag3.is_fallback is True
                assert diag3.resolved_lock_path.endswith(".atomic_lock")
                assert any("TIER2_FLOCK_FAILED" in r for r in diag3.fallback_reasons)
        finally:
            os.chmod(git_dir, 0o755)

        # Step 4: Tier 1 unwritable + Tier 2 fails + Tier 3 O_CREAT fails (EACCES) -> Tier 4
        real_open = os.open
        def mock_open_ro(path, flags, *args, **kwargs):
            if (flags & os.O_CREAT) or (flags & os.O_RDWR) or (flags & os.O_WRONLY):
                raise OSError(errno.EACCES, "Simulated EROFS/EACCES")
            return real_open(path, flags, *args, **kwargs)
        monkeypatch.setattr(os, "open", mock_open_ro)

        lock4 = WorktreeEventLock(repo, timeout_seconds=2.0)
        with lock4:
            diag4 = lock4.get_diagnostic()
            assert diag4.active_tier == LockTier.IN_MEMORY_MUTEX
            assert diag4.is_fallback is True
            assert diag4.resolved_lock_path == f"in_memory://{repo.resolve()}"

    def test_tier4_in_memory_mutex_concurrency_and_async(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Simulated complete EROFS sandbox maintains mutual exclusion across threads and async tasks in Tier 4."""
        repo = tmp_path / "in_mem_repo"
        repo.mkdir()

        # Force all disk writes to fail
        real_open = os.open
        def mock_open_ro(path, flags, *args, **kwargs):
            if (flags & os.O_CREAT) or (flags & os.O_RDWR) or (flags & os.O_WRONLY):
                raise OSError(errno.EROFS, "Read-only file system")
            return real_open(path, flags, *args, **kwargs)
        monkeypatch.setattr(os, "open", mock_open_ro)

        counter = 0
        lock_guard = threading.Lock()
        active_holders = 0
        overlap = False

        def thread_task():
            nonlocal counter, active_holders, overlap
            for _ in range(20):
                lock = WorktreeEventLock(repo, timeout_seconds=5.0)
                with lock:
                    with lock_guard:
                        active_holders += 1
                        if active_holders > 1:
                            overlap = True
                        c = counter
                    time.sleep(0.0005)
                    with lock_guard:
                        counter = c + 1
                        active_holders -= 1

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(thread_task) for _ in range(8)]
            for f in futures:
                f.result()

        assert not overlap
        assert counter == 8 * 20


# ===========================================================================
# 5. Nested Worktree .git Pointer Resolution & Traversal
# ===========================================================================

class TestResolveGitDirAdversarialDeep:
    """Stress tests for resolve_git_dir pointer chains, ascending traversal, and bare repos."""

    def test_deep_nested_worktree_pointer_chain(self, tmp_path: Path):
        """Chained worktree pointers: wt1 -> wt2 -> wt3 -> bare_git."""
        bare_repo = tmp_path / "main_bare.git"
        bare_repo.mkdir()
        (bare_repo / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (bare_repo / "objects").mkdir()
        (bare_repo / "refs").mkdir()

        wt3 = tmp_path / "wt3"
        wt3.mkdir()
        (wt3 / ".git").write_text(f"gitdir: {bare_repo.resolve()}\n", encoding="utf-8")

        wt2 = tmp_path / "wt2"
        wt2.mkdir()
        (wt2 / ".git").write_text(f"gitdir: {(wt3 / '.git').resolve()}\n", encoding="utf-8")

        wt1 = tmp_path / "wt1"
        wt1.mkdir()
        (wt1 / ".git").write_text(f"gitdir: {(wt2 / '.git').resolve()}\n", encoding="utf-8")

        resolved = resolve_git_dir(wt1)
        assert resolved.exists()

    def test_deep_directory_ascension_to_git_root(self, tmp_path: Path):
        """Ascends from a 15-level deep child directory to locate repo root .git."""
        repo = tmp_path / "root_repo"
        repo.mkdir()
        git_dir = repo / ".git"
        git_dir.mkdir()

        curr = repo
        for i in range(15):
            curr = curr / f"level_{i:02d}"
            curr.mkdir()

        resolved = resolve_git_dir(curr)
        assert resolved == git_dir.resolve()

    def test_circular_worktree_pointer_loop(self, tmp_path: Path):
        """Circular pointer loop A -> B -> A does not cause infinite recursion."""
        wtA = tmp_path / "wtA"
        wtA.mkdir()
        wtB = tmp_path / "wtB"
        wtB.mkdir()

        (wtA / ".git").write_text(f"gitdir: {(wtB / '.git').resolve()}\n", encoding="utf-8")
        (wtB / ".git").write_text(f"gitdir: {(wtA / '.git').resolve()}\n", encoding="utf-8")

        resolvedA = resolve_git_dir(wtA)
        assert isinstance(resolvedA, Path)

    def test_bare_git_repo_resolution_with_subdirectories(self, tmp_path: Path):
        """Subdirectories within a bare repository resolve directly to the bare repo directory."""
        bare_repo = tmp_path / "my_bare.git"
        bare_repo.mkdir()
        (bare_repo / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (bare_repo / "objects").mkdir()
        (bare_repo / "refs").mkdir()

        sub = bare_repo / "objects" / "pack"
        sub.mkdir(parents=True)

        resolved = resolve_git_dir(sub)
        assert resolved == bare_repo.resolve()


# ===========================================================================
# 6. KnowledgeStore Link Fallback Matrix & SHA256 Integrity
# ===========================================================================

class TestKnowledgeStoreLinkFallbackAndIntegrity:
    """Adversarial stress tests for KnowledgeStore atomic publication and integrity validation."""

    @pytest.mark.parametrize("simulated_errno", [
        errno.EXDEV,
        errno.EPERM,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.EPERM),
        getattr(errno, "EOPNOTSUPP", errno.EPERM),
        getattr(errno, "EMLINK", errno.EPERM),
    ])
    def test_all_link_fallback_errnos_matrix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, simulated_errno: int):
        """All supported link failure errnos seamlessly fall back to atomic os.replace."""
        store = KnowledgeStore(tmp_path / f"store_errno_{simulated_errno}")

        def mock_link(src: Path | str, dst: Path | str):
            raise OSError(simulated_errno, f"Simulated link errno {simulated_errno}")

        monkeypatch.setattr(os, "link", mock_link)

        content = f"Payload resilience test for errno {simulated_errno}."
        record = store.upsert_document(
            document_id=f"doc_{simulated_errno}",
            folder="resilience/matrix",
            title=f"Doc {simulated_errno}",
            content=content,
        )

        assert record.document_id == f"doc_{simulated_errno}"
        source_file = store.root / record.relative_path
        assert source_file.exists()
        assert source_file.read_text(encoding="utf-8") == content

        # Check zero tmp files left
        versions_dir = store.root / "sources" / "resilience" / "matrix" / f"doc_{simulated_errno}" / "versions"
        tmp_files = list(versions_dir.glob(".*.tmp"))
        assert len(tmp_files) == 0

    def test_link_and_replace_both_failing_cleans_temporary_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """When both link and replace fail (e.g. ENOSPC), temporary file is guaranteed to be deleted."""
        store = KnowledgeStore(tmp_path / "store_double_fail")

        def mock_link(src, dst):
            raise OSError(errno.EXDEV, "Cross-device link")

        def mock_replace(src, dst):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(os, "link", mock_link)
        monkeypatch.setattr(os, "replace", mock_replace)

        with pytest.raises(OSError) as exc_info:
            store.upsert_document("doc_nospc", "storage", "Title", "Some document content")
        assert exc_info.value.errno == errno.ENOSPC

        # Verify no .tmp files leaked
        versions_dir = store.root / "sources" / "storage" / "doc_nospc" / "versions"
        if versions_dir.exists():
            tmp_files = list(versions_dir.glob(".*.tmp"))
            assert len(tmp_files) == 0

    def test_tampered_source_file_sha256_detection(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Tampering with blob bytes between write and validation triggers KnowledgeStoreError."""
        store = KnowledgeStore(tmp_path / "store_tamper")

        real_link = os.link
        def mock_tampering_link(src: Path | str, dst: Path | str):
            # Tamper with file before link
            Path(src).write_bytes(b"TAMPERED_BYTE_PAYLOAD")
            real_link(src, dst)

        monkeypatch.setattr(os, "link", mock_tampering_link)

        with pytest.raises(KnowledgeStoreError, match="content-addressed source file has been modified"):
            store.upsert_document(
                document_id="tamper_doc",
                folder="security",
                title="Tampered Payload",
                content="Original uncorrupted content.",
            )


# ===========================================================================
# 7. KnowledgeStore High Concurrency & Revision Conflict Handling
# ===========================================================================

class TestKnowledgeStoreConcurrentPublications:
    """Stress tests for multi-process upserts, revision conflicts, and WAL consistency."""

    def test_concurrent_upsert_distinct_documents_multiprocess(self, tmp_path: Path):
        """10 OS processes concurrently upserting 20 distinct documents under simulated EXDEV."""
        store_root = str((tmp_path / "mp_ks_store").resolve())
        num_docs = 20

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes = []

        for i in range(num_docs):
            p = ctx.Process(
                target=_mp_ks_upsert_worker,
                args=(
                    store_root,
                    f"doc_{i:03d}",
                    f"folder_{i % 4}",
                    f"Title {i}",
                    f"Unique content for document {i} with uuid {uuid.uuid4().hex}.",
                    True,  # simulate EXDEV
                    queue,
                ),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=30.0)
            assert not p.is_alive()

        results = [queue.get() for _ in range(num_docs)]
        for pid, ok, res in results:
            assert ok is True, f"Process {pid} failed: {res}"

        store = KnowledgeStore(store_root)
        for i in range(num_docs):
            packet = store.retrieve(f"document {i}", top_k=5)
            assert len(packet.items) >= 1
            assert any(item.document_id == f"doc_{i:03d}" for item in packet.items)

    def test_concurrent_revision_conflict_handling(self, tmp_path: Path):
        """Multiple threads competing to update same revision: exactly 1 succeeds, others get DocumentVersionConflictError."""
        store = KnowledgeStore(tmp_path / "conflict_store")
        initial = store.upsert_document(
            document_id="shared_conflict_doc",
            folder="legal",
            title="Master Agreement",
            content="Initial revision content 1.0.",
        )

        num_threads = 10
        success_revisions: list[str] = []
        conflict_errors: list[str] = []
        lock = threading.Lock()

        def update_task(idx: int):
            try:
                rec = store.upsert_document(
                    document_id="shared_conflict_doc",
                    folder="legal",
                    title="Master Agreement",
                    content=f"Updated revision content variant {idx}.",
                    expected_revision_id=initial.revision_id,
                    revision_reason="content_update",
                )
                with lock:
                    success_revisions.append(rec.revision_id)
            except DocumentVersionConflictError as exc:
                with lock:
                    conflict_errors.append(str(exc))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(update_task, i) for i in range(num_threads)]
            for f in futures:
                f.result()

        # Exactly 1 thread successfully committed its update targeting initial revision
        assert len(success_revisions) == 1
        assert len(conflict_errors) == num_threads - 1

    def test_concurrent_idempotent_duplicate_upserts(self, tmp_path: Path):
        """20 concurrent threads calling upsert_document with identical content -> idempotent deduplication."""
        store = KnowledgeStore(tmp_path / "idempotent_store")
        doc_id = "dedup_doc"
        content = "Exact deterministic payload for deduplication testing."

        records: list[DocumentRecord] = []
        lock = threading.Lock()

        def insert_task():
            rec = store.upsert_document(
                document_id=doc_id,
                folder="ops",
                title="Standard SOP",
                content=content,
            )
            with lock:
                records.append(rec)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(insert_task) for _ in range(20)]
            for f in futures:
                f.result()

        assert len(records) == 20
        first_rev = records[0].revision_id
        assert all(r.revision_id == first_rev for r in records)

    def test_high_throughput_wal_mixed_reads_writes(self, tmp_path: Path):
        """High-throughput parallel mix of readers and writers operating concurrently under WAL mode."""
        store = KnowledgeStore(tmp_path / "wal_stress_store")

        # Seed initial documents
        for i in range(5):
            store.upsert_document(f"seed_{i}", "seed", f"Seed {i}", f"Initial seed content {i} for indexing.")

        stop_event = threading.Event()
        errors: list[Exception] = []
        reads_completed = 0
        writes_completed = 0
        stats_lock = threading.Lock()

        def reader_task(r_id: int):
            nonlocal reads_completed
            while not stop_event.is_set():
                try:
                    packet = store.retrieve("seed content", top_k=3)
                    assert packet.items is not None
                    with stats_lock:
                        reads_completed += 1
                except (OSError, RuntimeError, ValueError, KnowledgeStoreError) as exc:
                    with stats_lock:
                        errors.append(exc)
                time.sleep(0.002)

        def writer_task(w_id: int):
            nonlocal writes_completed
            for step in range(15):
                try:
                    store.upsert_document(
                        document_id=f"writer_{w_id}_{step}",
                        folder="dynamic",
                        title=f"Dynamic {w_id}-{step}",
                        content=f"Dynamically published content payload {w_id}_{step}_{uuid.uuid4().hex}.",
                    )
                    with stats_lock:
                        writes_completed += 1
                except (OSError, RuntimeError, ValueError, KnowledgeStoreError) as exc:
                    with stats_lock:
                        errors.append(exc)
                time.sleep(0.005)

        readers = [threading.Thread(target=reader_task, args=(i,)) for i in range(6)]
        writers = [threading.Thread(target=writer_task, args=(i,)) for i in range(4)]

        for r in readers:
            r.start()
        for w in writers:
            w.start()

        for w in writers:
            w.join(timeout=25.0)

        stop_event.set()
        for r in readers:
            r.join(timeout=10.0)

        assert len(errors) == 0, f"Errors observed during WAL stress: {errors}"
        assert writes_completed == 4 * 15
        assert reads_completed > 20
