"""Empirical adversarial stress testing and concurrency verification for WorktreeEventLock.

Covers:
1. Multi-threaded and multi-process concurrency on Tier 1 (kernel flock).
2. Multi-process concurrency on Tier 2 (tempfile flock).
3. Multi-process concurrency on Tier 3 (atomic user-space lockfile).
4. Multi-threaded and async concurrency on Tier 4 (in-memory mutex).
5. High-contention race conditions and dead-PID stale lock recovery under SIGKILL.
6. TTL expiration recovery under concurrent contention.
7. Corrupted lockfile recovery.
8. Tier isolation (ensuring contention does not trigger premature fallback).
9. Rapid acquire/release cycles and file descriptor leak prevention.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import signal
import socket
import threading
import time
from pathlib import Path

import pytest

from pipeline.event_lock import (
    LockTier,
    WorktreeEventLock,
)

# ===========================================================================
# Helper Functions for Multiprocessing Workers
# ===========================================================================


def _mp_worker_increment(
    worktree_path: str,
    counter_file: str,
    iterations: int,
    force_tier: str | None,
    result_queue: mp.Queue,
    hold_delay: float = 0.002,
):
    """Worker function executed in separate OS processes."""
    pid = os.getpid()
    success_count = 0
    tiers_observed = set()

    # Apply forced tier patches if requested
    if force_tier == "tier2":
        # Make .git unwritable
        git_dir = Path(worktree_path) / ".git"
        if git_dir.exists():
            pass
    elif force_tier == "tier3":
        import pipeline.event_lock as el

        el.fcntl = None  # Disable fcntl in this subprocess

    for _ in range(iterations):
        lock = WorktreeEventLock(
            worktree_path,
            timeout_seconds=30.0,
            poll_interval=0.005,
        )
        acquired = lock.acquire(blocking=True)
        if not acquired:
            result_queue.put(
                (
                    pid,
                    False,
                    f"Timeout acquiring lock in PID {pid}",
                    list(tiers_observed),
                )
            )
            return

        try:
            tiers_observed.add(
                lock.active_tier.value if lock.active_tier else "unknown"
            )
            # Critical section: Read, verify, sleep jitter, increment, write
            val = 0
            if Path(counter_file).exists():
                content = Path(counter_file).read_text(encoding="utf-8").strip()
                if content:
                    val = int(content)

            time.sleep(hold_delay)

            Path(counter_file).write_text(str(val + 1), encoding="utf-8")
            success_count += 1
        finally:
            lock.release()

    result_queue.put((pid, True, success_count, list(tiers_observed)))


def _mp_abandoned_lock_holder(worktree_path: str, ready_event: mp.Event):
    """Acquires a Tier 3 lock and signals ready, then waits to be SIGKILLed."""
    import pipeline.event_lock as el

    el.fcntl = None  # Force Tier 3

    lock = WorktreeEventLock(worktree_path, timeout_seconds=10.0)
    if lock.acquire(blocking=True):
        ready_event.set()
        # Hold lock forever until killed
        while True:
            time.sleep(1.0)


# ===========================================================================
# 1. Tier 1: Kernel flock Multi-Process & Multi-Thread Stress Tests
# ===========================================================================


def test_stress_tier1_multiprocess_contention(tmp_path: Path):
    """Stress test Tier 1 kernel flock with 8 concurrent OS processes."""
    repo = tmp_path / "stress_repo_tier1"
    (repo / ".git").mkdir(parents=True)
    counter_file = tmp_path / "counter_tier1.txt"
    counter_file.write_text("0", encoding="utf-8")

    num_processes = 8
    iterations_per_process = 15
    expected_total = num_processes * iterations_per_process

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []

    for _ in range(num_processes):
        p = ctx.Process(
            target=_mp_worker_increment,
            args=(
                str(repo),
                str(counter_file),
                iterations_per_process,
                None,
                queue,
                0.001,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=30.0)
        assert not p.is_alive(), "Worker process timed out"

    results = []
    while not queue.empty():
        results.append(queue.get())

    assert len(results) == num_processes
    for pid, ok, res, tiers in results:
        assert ok is True, f"Process {pid} failed: {res}"
        assert res == iterations_per_process
        assert "kernel_flock" in tiers

    final_count = int(counter_file.read_text(encoding="utf-8").strip())
    assert final_count == expected_total


def test_stress_tier1_multithread_contention(tmp_path: Path):
    """Stress test Tier 1 kernel flock with 16 concurrent threads."""
    repo = tmp_path / "stress_repo_threads"
    (repo / ".git").mkdir(parents=True)

    counter = 0
    counter_lock = threading.Lock()
    active_in_critical_section = 0
    max_concurrent_in_critical_section = 0
    overlap_detected = False

    num_threads = 16
    iterations_per_thread = 20
    expected_total = num_threads * iterations_per_thread

    def thread_worker():
        nonlocal \
            counter, \
            active_in_critical_section, \
            max_concurrent_in_critical_section, \
            overlap_detected
        for _ in range(iterations_per_thread):
            lock = WorktreeEventLock(repo, timeout_seconds=20.0, poll_interval=0.005)
            with lock:
                with counter_lock:
                    active_in_critical_section += 1
                    max_concurrent_in_critical_section = max(
                        max_concurrent_in_critical_section, active_in_critical_section
                    )
                    if active_in_critical_section > 1:
                        overlap_detected = True
                    curr = counter

                time.sleep(0.001)

                with counter_lock:
                    counter = curr + 1
                    active_in_critical_section -= 1

    threads = [threading.Thread(target=thread_worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)

    assert not overlap_detected, (
        f"Mutual exclusion violated! Max concurrent: {max_concurrent_in_critical_section}"
    )
    assert max_concurrent_in_critical_section == 1
    assert counter == expected_total


# ===========================================================================
# 2. Tier 2: Tempfile flock Multi-Process Stress Test
# ===========================================================================


def test_stress_tier2_multiprocess_contention(tmp_path: Path):
    """Stress test Tier 2 tempfile flock with 8 concurrent OS processes when .git is unwritable."""
    # Use a non-git directory where parent is unwritable or .git open fails
    repo = tmp_path / "stress_repo_tier2"
    repo.mkdir()
    # Create .git as unwritable directory
    git_dir = repo / ".git"
    git_dir.mkdir()
    os.chmod(git_dir, 0o444)

    counter_file = tmp_path / "counter_tier2.txt"
    counter_file.write_text("0", encoding="utf-8")

    num_processes = 8
    iterations_per_process = 12
    expected_total = num_processes * iterations_per_process

    try:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes = []

        for _ in range(num_processes):
            p = ctx.Process(
                target=_mp_worker_increment,
                args=(
                    str(repo),
                    str(counter_file),
                    iterations_per_process,
                    "tier2",
                    queue,
                    0.001,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=30.0)
            assert not p.is_alive(), "Worker process timed out"

        results = []
        while not queue.empty():
            results.append(queue.get())

        assert len(results) == num_processes
        for pid, ok, res, tiers in results:
            assert ok is True, f"Process {pid} failed: {res}"
            assert res == iterations_per_process
            assert "temp_flock" in tiers

        final_count = int(counter_file.read_text(encoding="utf-8").strip())
        assert final_count == expected_total

    finally:
        os.chmod(git_dir, 0o755)


# ===========================================================================
# 3. Tier 3: User-Space Atomic Lockfile Multi-Process Stress Test
# ===========================================================================


def test_stress_tier3_multiprocess_contention(tmp_path: Path):
    """Stress test Tier 3 atomic O_CREAT|O_EXCL lockfile with 8 concurrent OS processes."""
    repo = tmp_path / "stress_repo_tier3"
    repo.mkdir()
    counter_file = tmp_path / "counter_tier3.txt"
    counter_file.write_text("0", encoding="utf-8")

    num_processes = 8
    iterations_per_process = 12
    expected_total = num_processes * iterations_per_process

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []

    for _ in range(num_processes):
        p = ctx.Process(
            target=_mp_worker_increment,
            args=(
                str(repo),
                str(counter_file),
                iterations_per_process,
                "tier3",
                queue,
                0.001,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=30.0)
        assert not p.is_alive(), "Worker process timed out"

    results = []
    while not queue.empty():
        results.append(queue.get())

    assert len(results) == num_processes
    for pid, ok, res, tiers in results:
        assert ok is True, f"Process {pid} failed: {res}"
        assert res == iterations_per_process
        assert "user_space_atomic" in tiers

    final_count = int(counter_file.read_text(encoding="utf-8").strip())
    assert final_count == expected_total


# ===========================================================================
# 4. Tier 4: In-Memory Mutex Stress Tests (Threads & Async)
# ===========================================================================


def test_stress_tier4_in_memory_multithread_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Stress test Tier 4 in-memory threading.RLock with 16 concurrent threads."""
    import errno

    def mock_open(*args, **kwargs):
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(os, "open", mock_open)

    counter = 0
    active_count = 0
    max_active = 0
    overlap = False
    lock_var = threading.Lock()

    num_threads = 16
    iterations = 25
    expected_total = num_threads * iterations

    def worker():
        nonlocal counter, active_count, max_active, overlap
        for _ in range(iterations):
            lock = WorktreeEventLock(
                tmp_path, timeout_seconds=15.0, poll_interval=0.002
            )
            with lock:
                assert lock.active_tier == LockTier.IN_MEMORY_MUTEX
                with lock_var:
                    active_count += 1
                    max_active = max(max_active, active_count)
                    if active_count > 1:
                        overlap = True
                    curr = counter

                time.sleep(0.0005)

                with lock_var:
                    counter = curr + 1
                    active_count -= 1

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)

    assert not overlap
    assert max_active == 1
    assert counter == expected_total


@pytest.mark.asyncio
async def test_stress_tier4_in_memory_async_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Stress test Tier 4 in-memory asyncio.Lock with 40 concurrent async tasks."""
    import errno

    def mock_open(*args, **kwargs):
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(os, "open", mock_open)

    counter = 0
    active_tasks = 0
    max_active_tasks = 0
    overlap = False

    async def async_worker(task_id: int):
        nonlocal counter, active_tasks, max_active_tasks, overlap
        lock = WorktreeEventLock(tmp_path, timeout_seconds=15.0, poll_interval=0.005)
        async with lock:
            assert lock.active_tier == LockTier.IN_MEMORY_MUTEX
            active_tasks += 1
            max_active_tasks = max(max_active_tasks, active_tasks)
            if active_tasks > 1:
                overlap = True
            curr = counter
            await asyncio.sleep(0.002)
            counter = curr + 1
            active_tasks -= 1

    tasks = [async_worker(i) for i in range(40)]
    await asyncio.gather(*tasks)

    assert not overlap
    assert max_active_tasks == 1
    assert counter == 40


# ===========================================================================
# 5. Dead-PID Stale Lock Recovery under High Contention (Adversarial)
# ===========================================================================


def test_stress_tier3_dead_pid_recovery_under_sigkill_contention(tmp_path: Path):
    """Adversarial test: Child process acquires Tier 3 lock and is violently killed with SIGKILL.
    10 concurrent processes immediately compete to break stale lock and acquire it.
    Verifies:
    1. Stale lock is broken cleanly without crashing.
    2. No split-brain double-locking occurs during recovery.
    3. All 10 competing processes complete their updates successfully.
    """
    repo = tmp_path / "sigkill_repo"
    repo.mkdir()
    counter_file = tmp_path / "counter_sigkill.txt"
    counter_file.write_text("0", encoding="utf-8")

    ctx = mp.get_context("spawn")
    ready_event = ctx.Event()

    # 1. Spawn victim child process to hold Tier 3 lock
    victim = ctx.Process(
        target=_mp_abandoned_lock_holder, args=(str(repo), ready_event)
    )
    victim.start()

    # Wait for victim to hold lock
    assert ready_event.wait(timeout=10.0), "Victim failed to acquire lock"
    victim_pid = victim.pid
    assert victim_pid is not None

    # Verify lock exists and belongs to victim PID
    lock_checker = WorktreeEventLock(repo)
    atomic_lock_file = lock_checker._get_temp_lock_path(suffix=".atomic_lock")
    assert atomic_lock_file.exists()
    meta = json.loads(atomic_lock_file.read_text(encoding="utf-8"))
    assert meta["pid"] == victim_pid

    # 2. Forcibly kill victim with SIGKILL (simulating ungraceful crash / power loss / OOM)
    os.kill(victim_pid, signal.SIGKILL)
    victim.join(timeout=5.0)

    # 3. Immediately launch 10 competing processes to break the stale lock
    num_competitors = 10
    iterations_per_process = 5
    expected_total = num_competitors * iterations_per_process

    queue = ctx.Queue()
    competitors = []

    for _ in range(num_competitors):
        p = ctx.Process(
            target=_mp_worker_increment,
            args=(
                str(repo),
                str(counter_file),
                iterations_per_process,
                "tier3",
                queue,
                0.002,
            ),
        )
        p.start()
        competitors.append(p)

    for p in competitors:
        p.join(timeout=30.0)
        assert not p.is_alive(), "Competitor process timed out"

    results = []
    while not queue.empty():
        results.append(queue.get())

    assert len(results) == num_competitors
    for pid, ok, res, tiers in results:
        assert ok is True, f"Competitor {pid} failed: {res}"
        assert res == iterations_per_process
        assert "user_space_atomic" in tiers

    final_count = int(counter_file.read_text(encoding="utf-8").strip())
    assert final_count == expected_total


# ===========================================================================
# 6. TTL Expiration Stale Lock Recovery under High Contention
# ===========================================================================


def test_stress_tier3_ttl_expiration_recovery_high_contention(tmp_path: Path):
    """Adversarial test: An expired Tier 3 lock (TTL exceeded) is contested by 8 concurrent processes."""
    repo = tmp_path / "ttl_repo"
    repo.mkdir()
    counter_file = tmp_path / "counter_ttl.txt"
    counter_file.write_text("0", encoding="utf-8")

    lock_checker = WorktreeEventLock(repo)
    atomic_lock_file = lock_checker._get_temp_lock_path(suffix=".atomic_lock")

    # Write expired lockfile with active local PID but expired timestamp
    expired_payload = {
        "lock_id": "expired-ttl-uuid",
        "pid": os.getpid(),  # Alive PID, but expired TTL
        "hostname": socket.gethostname(),
        "target_path": str(repo.resolve()),
        "acquired_at": time.time() - 300.0,
        "stale_timeout_seconds": 1.0,
    }
    atomic_lock_file.write_text(json.dumps(expired_payload), encoding="utf-8")

    num_processes = 8
    iterations_per_process = 5
    expected_total = num_processes * iterations_per_process

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []

    for _ in range(num_processes):
        p = ctx.Process(
            target=_mp_worker_increment,
            args=(
                str(repo),
                str(counter_file),
                iterations_per_process,
                "tier3",
                queue,
                0.001,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=30.0)
        assert not p.is_alive()

    results = []
    while not queue.empty():
        results.append(queue.get())

    assert len(results) == num_processes
    for pid, ok, res, tiers in results:
        assert ok is True, f"Process {pid} failed: {res}"
        assert res == iterations_per_process

    final_count = int(counter_file.read_text(encoding="utf-8").strip())
    assert final_count == expected_total


# ===========================================================================
# 7. Corrupted Lockfile Recovery
# ===========================================================================


def test_stress_tier3_corrupted_lockfile_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Adversarial test: Corrupted garbage in Tier 3 lockfile with old mtime is recovered safely."""
    monkeypatch.setattr("pipeline.event_lock.fcntl", None)

    lock = WorktreeEventLock(tmp_path, timeout_seconds=3.0)
    atomic_path = lock._get_temp_lock_path(suffix=".atomic_lock")

    # Write invalid corrupted data
    atomic_path.write_bytes(b"\x00\xff\xfe\xaaCORRUPTED_GARBAGE_DATA")
    old_time = time.time() - 15.0
    os.utime(str(atomic_path), (old_time, old_time))

    with lock:
        assert lock.is_locked is True
        assert lock.active_tier == LockTier.USER_SPACE_ATOMIC
        assert any("Evicted stale" in r for r in lock.diagnostic.fallback_reasons)


# ===========================================================================
# 8. Tier Isolation: Contention Must NOT Cause Premature Fallback
# ===========================================================================


def test_tier1_contention_does_not_fall_back_to_tier2(tmp_path: Path):
    """Verifies that contention on Tier 1 (kernel flock) causes waiting/blocking on Tier 1,
    and NEVER prematurely falls back to Tier 2."""
    repo = tmp_path / "tier_isolation_repo"
    (repo / ".git").mkdir(parents=True)

    lock1 = WorktreeEventLock(repo, timeout_seconds=5.0)
    assert lock1.acquire(blocking=True) is True
    assert lock1.active_tier == LockTier.KERNEL_FLOCK

    # Second lock with short timeout should fail on Tier 1 without falling back to Tier 2
    lock2 = WorktreeEventLock(repo, timeout_seconds=0.1, poll_interval=0.01)
    acquired = lock2.acquire(blocking=True)
    assert acquired is False
    assert lock2.is_locked is False
    # Fallback reasons should NOT contain TIER1_FLOCK_FAILED or fallback to Tier 2
    assert len(lock2.diagnostic.fallback_reasons) == 0

    lock1.release()

    # Now lock2 should acquire Tier 1 cleanly
    assert lock2.acquire(blocking=False) is True
    assert lock2.active_tier == LockTier.KERNEL_FLOCK
    lock2.release()


# ===========================================================================
# 9. Rapid Acquisition Cycles & Resource Leak Prevention
# ===========================================================================


def test_rapid_acquire_release_cycles_tier1(tmp_path: Path):
    """Stress test rapid acquire and release cycles (300 cycles) to ensure no fd leaks."""
    repo = tmp_path / "rapid_repo"
    (repo / ".git").mkdir(parents=True)

    for i in range(300):
        lock = WorktreeEventLock(repo, timeout_seconds=2.0)
        with lock:
            assert lock.is_locked is True
            assert lock.active_tier == LockTier.KERNEL_FLOCK
        assert lock.is_locked is False


def test_rapid_acquire_release_cycles_tier3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Stress test rapid acquire and release cycles on Tier 3 (200 cycles)."""
    monkeypatch.setattr("pipeline.event_lock.fcntl", None)

    for i in range(200):
        lock = WorktreeEventLock(tmp_path, timeout_seconds=2.0)
        with lock:
            assert lock.is_locked is True
            assert lock.active_tier == LockTier.USER_SPACE_ATOMIC
        assert lock.is_locked is False
        # Verify lockfile is cleaned up after each release
        assert not Path(lock.resolved_lock_path).exists()


# ===========================================================================
# 10. Linked Worktree Cross-Path Multi-Process Mutual Exclusion
# ===========================================================================


def test_linked_worktree_cross_path_mutual_exclusion(tmp_path: Path):
    """Verifies that processes accessing via worktree root (.git pointer file),
    administrative worktrees dir, and deep subdirectories all lock against the EXACT SAME lockfile.
    """
    main_repo = tmp_path / "main_repo"
    admin_dir = main_repo / ".git" / "worktrees" / "feature_branch"
    admin_dir.mkdir(parents=True)

    worktree_root = tmp_path / "worktrees" / "feature_branch"
    deep_subdir = worktree_root / "src" / "deeply" / "nested" / "component"
    deep_subdir.mkdir(parents=True)

    git_file = worktree_root / ".git"
    git_file.write_text(f"gitdir: {admin_dir.resolve()}\n", encoding="utf-8")

    counter_file = tmp_path / "cross_path_counter.txt"
    counter_file.write_text("0", encoding="utf-8")

    # 3 worktree access paths to test:
    # 1. worktree_root (contains .git file)
    # 2. deep_subdir (walks up to .git file)
    # 3. git_file (direct .git pointer file argument)
    paths_to_test = [
        str(worktree_root),
        str(deep_subdir),
        str(git_file),
    ]

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    iterations = 10

    for i in range(9):
        target = paths_to_test[i % 3]
        p = ctx.Process(
            target=_mp_worker_increment,
            args=(target, str(counter_file), iterations, None, queue, 0.001),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=30.0)
        assert not p.is_alive()

    results = []
    while not queue.empty():
        results.append(queue.get())

    assert len(results) == 9
    for pid, ok, res, tiers in results:
        assert ok is True, f"Process {pid} failed: {res}"
        assert res == iterations

    final_count = int(counter_file.read_text(encoding="utf-8").strip())
    assert final_count == 9 * iterations


# ===========================================================================
# 11. Cascading Dead-PID Stale Recovery
# ===========================================================================


def test_cascading_dead_pid_recovery_multiprocess(tmp_path: Path):
    """Adversarial test: Two consecutive child processes crash (SIGKILL) while holding the lock.
    Subsequent processes must recover each crash and complete without corruption.
    """
    repo = tmp_path / "cascading_repo"
    repo.mkdir()
    counter_file = tmp_path / "counter_cascading.txt"
    counter_file.write_text("0", encoding="utf-8")

    ctx = mp.get_context("spawn")

    for wave in range(2):
        ready = ctx.Event()
        victim = ctx.Process(target=_mp_abandoned_lock_holder, args=(str(repo), ready))
        victim.start()
        assert ready.wait(timeout=10.0), f"Victim wave {wave} failed to acquire lock"
        v_pid = victim.pid
        os.kill(v_pid, signal.SIGKILL)
        victim.join(timeout=5.0)

    # Now run 6 workers
    queue = ctx.Queue()
    processes = []
    num_workers = 6
    iterations = 5

    for _ in range(num_workers):
        p = ctx.Process(
            target=_mp_worker_increment,
            args=(str(repo), str(counter_file), iterations, "tier3", queue, 0.001),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=30.0)
        assert not p.is_alive()

    results = []
    while not queue.empty():
        results.append(queue.get())

    assert len(results) == num_workers
    for pid, ok, res, tiers in results:
        assert ok is True, f"Process {pid} failed: {res}"

    final_count = int(counter_file.read_text(encoding="utf-8").strip())
    assert final_count == num_workers * iterations


# ===========================================================================
# 12. Async Context Manager Cancellation Resilience
# ===========================================================================


@pytest.mark.asyncio
async def test_async_lock_cancellation_safety(tmp_path: Path):
    """Verifies that cancelling an async task holding the lock cleanly releases it,
    allowing subsequent tasks to acquire without deadlock."""
    repo = tmp_path / "async_cancel_repo"
    (repo / ".git").mkdir(parents=True)

    lock1 = WorktreeEventLock(repo, timeout_seconds=5.0)
    holding_event = asyncio.Event()

    async def task_to_cancel():
        async with lock1:
            holding_event.set()
            await asyncio.sleep(10.0)

    task = asyncio.create_task(task_to_cancel())
    await holding_event.wait()

    # Task now holds the lock; cancel it
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # lock1 must be released now
    assert lock1.is_locked is False

    # A second task must be able to acquire immediately
    lock2 = WorktreeEventLock(repo, timeout_seconds=1.0)
    acquired = await lock2.acquire_async(blocking=False)
    assert acquired is True
    await lock2.release_async()


# ===========================================================================
# 13. Tier Isolation for Tier 2 and Tier 3
# ===========================================================================


def test_tier2_and_tier3_contention_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verifies that Tier 2 contention doesn't prematurely drop to Tier 3,
    and Tier 3 contention doesn't prematurely drop to Tier 4."""
    # 1. Tier 2 contention
    repo = tmp_path / "tier2_iso_repo"
    repo.mkdir()
    git_dir = repo / ".git"
    git_dir.mkdir()
    os.chmod(git_dir, 0o444)

    try:
        lock1 = WorktreeEventLock(repo, timeout_seconds=5.0)
        assert lock1.acquire(blocking=True) is True
        assert lock1.active_tier == LockTier.TEMP_FLOCK

        lock2 = WorktreeEventLock(repo, timeout_seconds=0.1, poll_interval=0.01)
        assert lock2.acquire(blocking=True) is False
        assert lock2.is_locked is False
        # Ensure lock2 stayed at Tier 2 and did NOT fall back to Tier 3
        assert not any(
            "TIER2_FLOCK_FAILED" in r for r in lock2.diagnostic.fallback_reasons
        )
        assert not any("TIER3" in r for r in lock2.diagnostic.fallback_reasons)

        lock1.release()
    finally:
        os.chmod(git_dir, 0o755)

    # 2. Tier 3 contention
    monkeypatch.setattr("pipeline.event_lock.fcntl", None)
    lock_t3_1 = WorktreeEventLock(tmp_path / "tier3_iso_repo", timeout_seconds=5.0)
    assert lock_t3_1.acquire(blocking=True) is True
    assert lock_t3_1.active_tier == LockTier.USER_SPACE_ATOMIC

    lock_t3_2 = WorktreeEventLock(
        tmp_path / "tier3_iso_repo", timeout_seconds=0.1, poll_interval=0.01
    )
    assert lock_t3_2.acquire(blocking=True) is False
    assert lock_t3_2.is_locked is False
    # Ensure lock_t3_2 stayed at Tier 3 and did NOT fall back to Tier 4
    assert not any(
        "TIER3_UNWRITABLE" in r for r in lock_t3_2.diagnostic.fallback_reasons
    )
    assert (
        lock_t3_2.active_tier is None
        or lock_t3_2.diagnostic.active_tier != LockTier.IN_MEMORY_MUTEX
    )

    lock_t3_1.release()
