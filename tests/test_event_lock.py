"""Comprehensive unit and integration tests for WorktreeEventLock and resolve_git_dir."""
from __future__ import annotations

import asyncio
import errno
import json
import os
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest

from pipeline.event_lock import (
    LockDiagnostic,
    LockTier,
    WorktreeEventLock,
    resolve_git_dir,
)

# ===========================================================================
# 1. resolve_git_dir Unit Tests
# ===========================================================================

def test_resolve_git_dir_standard_repo(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    git_dir = repo_dir / ".git"
    git_dir.mkdir()

    resolved = resolve_git_dir(repo_dir)
    assert resolved == git_dir.resolve()


def test_resolve_git_dir_linked_worktree_absolute(tmp_path: Path):
    main_repo = tmp_path / "main_repo"
    main_git_worktrees = main_repo / ".git" / "worktrees" / "wt1"
    main_git_worktrees.mkdir(parents=True)

    worktree_dir = tmp_path / "wt1"
    worktree_dir.mkdir()
    git_file = worktree_dir / ".git"
    git_file.write_text(f"gitdir: {main_git_worktrees.resolve()}\n", encoding="utf-8")

    resolved = resolve_git_dir(worktree_dir)
    assert resolved == main_git_worktrees.resolve()


def test_resolve_git_dir_linked_worktree_relative(tmp_path: Path):
    main_repo = tmp_path / "main_repo"
    main_git_worktrees = main_repo / ".git" / "worktrees" / "wt2"
    main_git_worktrees.mkdir(parents=True)

    worktree_dir = tmp_path / "worktrees_dir" / "wt2"
    worktree_dir.mkdir(parents=True)
    git_file = worktree_dir / ".git"
    git_file.write_text("gitdir: ../../main_repo/.git/worktrees/wt2\n", encoding="utf-8")

    resolved = resolve_git_dir(worktree_dir)
    assert resolved == main_git_worktrees.resolve()


def test_resolve_git_dir_subdirectory_walk(tmp_path: Path):
    worktree_dir = tmp_path / "worktree"
    sub_dir = worktree_dir / "src" / "pipeline" / "nested"
    sub_dir.mkdir(parents=True)

    git_target = tmp_path / "admin" / "git_dir"
    git_target.mkdir(parents=True)
    git_file = worktree_dir / ".git"
    git_file.write_text(f"gitdir: {git_target.resolve()}\n", encoding="utf-8")

    resolved = resolve_git_dir(sub_dir)
    assert resolved == git_target.resolve()


def test_resolve_git_dir_bare_repo(tmp_path: Path):
    bare_repo = tmp_path / "bare.git"
    bare_repo.mkdir()
    (bare_repo / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (bare_repo / "objects").mkdir()
    (bare_repo / "refs").mkdir()

    resolved = resolve_git_dir(bare_repo)
    assert resolved == bare_repo.resolve()


def test_resolve_git_dir_non_git_directory(tmp_path: Path):
    non_git = tmp_path / "plain_folder"
    non_git.mkdir()

    resolved = resolve_git_dir(non_git)
    assert resolved == non_git.resolve()


def test_resolve_git_dir_corrupted_pointer_file(tmp_path: Path):
    worktree_dir = tmp_path / "corrupt_worktree"
    worktree_dir.mkdir()
    git_file = worktree_dir / ".git"
    git_file.write_text("corrupted content without gitdir prefix\n", encoding="utf-8")

    resolved = resolve_git_dir(worktree_dir)
    assert resolved == worktree_dir.resolve()


def test_resolve_git_dir_direct_git_file_argument(tmp_path: Path):
    target_admin = tmp_path / "admin"
    target_admin.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    git_file = worktree / ".git"
    git_file.write_text(f"gitdir: {target_admin.resolve()}\n", encoding="utf-8")

    resolved = resolve_git_dir(git_file)
    assert resolved == target_admin.resolve()


def test_resolve_git_dir_direct_git_dir_argument(tmp_path: Path):
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    git_dir.mkdir(parents=True)

    resolved = resolve_git_dir(git_dir)
    assert resolved == git_dir.resolve()


# ===========================================================================
# 2. Tier 1: KERNEL_FLOCK Tests
# ===========================================================================

def test_tier1_kernel_flock_basic_acquire_and_release(tmp_path: Path):
    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    git_dir.mkdir(parents=True)

    lock = WorktreeEventLock(repo, timeout_seconds=2.0)
    assert not lock.is_locked

    acquired = lock.acquire(blocking=True)
    assert acquired is True
    assert lock.is_locked is True
    assert lock.active_tier == LockTier.KERNEL_FLOCK
    assert lock.resolved_lock_path == str(git_dir / "worktree_event.lock")
    assert Path(lock.resolved_lock_path).exists()

    diag = lock.get_diagnostic()
    assert diag.active_tier == LockTier.KERNEL_FLOCK
    assert diag.is_locked is True
    assert diag.holder_pid == os.getpid()
    assert diag.is_fallback is False

    lock.release()
    assert not lock.is_locked
    assert lock.get_diagnostic().is_locked is False


def test_tier1_kernel_flock_context_manager(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)

    with WorktreeEventLock(repo) as lock:
        assert lock.is_locked is True
        assert lock.active_tier == LockTier.KERNEL_FLOCK
        assert lock.diagnostic().is_locked is True

    assert lock.is_locked is False


def test_tier1_kernel_flock_contention_non_blocking(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)

    lock1 = WorktreeEventLock(repo, timeout_seconds=1.0)
    lock2 = WorktreeEventLock(repo, timeout_seconds=0.1)

    assert lock1.acquire(blocking=True) is True
    assert lock2.acquire(blocking=False) is False

    lock1.release()
    assert lock2.acquire(blocking=False) is True
    lock2.release()


def test_tier1_kernel_flock_contention_threads(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)

    execution_order = []

    def worker(worker_id: int, hold_seconds: float):
        lock = WorktreeEventLock(repo, timeout_seconds=5.0, poll_interval=0.01)
        with lock:
            execution_order.append(f"start_{worker_id}")
            time.sleep(hold_seconds)
            execution_order.append(f"end_{worker_id}")

    t1 = threading.Thread(target=worker, args=(1, 0.1))
    t2 = threading.Thread(target=worker, args=(2, 0.1))

    t1.start()
    time.sleep(0.02)
    t2.start()

    t1.join()
    t2.join()

    assert execution_order == ["start_1", "end_1", "start_2", "end_2"]


# ===========================================================================
# 3. Tier 2: TEMP_FLOCK Tests
# ===========================================================================

def test_tier2_temp_flock_fallback_on_unwritable_git_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = tmp_path / "readonly_repo"
    git_dir = repo / ".git"
    git_dir.mkdir(parents=True)

    # Force Tier 1 open on git_dir to fail with PermissionError
    orig_open = os.open

    def mock_open(path, flags, *args, **kwargs):
        if str(git_dir) in str(path):
            raise PermissionError(errno.EACCES, "Permission denied on .git")
        return orig_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", mock_open)

    lock = WorktreeEventLock(repo, timeout_seconds=2.0)
    with lock:
        assert lock.active_tier == LockTier.TEMP_FLOCK
        assert lock.diagnostic.is_fallback is True
        assert "TIER1_PATH_ERROR" in lock.diagnostic.fallback_reasons[0] or "TIER1_OPEN_FAILED" in lock.diagnostic.fallback_reasons[0]
        assert Path(lock.resolved_lock_path).parent == Path(tempfile.gettempdir())


def test_tier2_hash_uniqueness_and_determinism(tmp_path: Path):
    path_a = tmp_path / "project_alpha"
    path_b = tmp_path / "project_beta"
    path_a.mkdir()
    path_b.mkdir()

    lock_a1 = WorktreeEventLock(path_a)
    lock_a2 = WorktreeEventLock(path_a)
    lock_b = WorktreeEventLock(path_b)

    temp_a1 = lock_a1._get_temp_lock_path()
    temp_a2 = lock_a2._get_temp_lock_path()
    temp_b = lock_b._get_temp_lock_path()

    assert temp_a1 == temp_a2
    assert temp_a1 != temp_b
    assert "project_alpha" in temp_a1.name
    assert "project_beta" in temp_b.name


# ===========================================================================
# 4. Tier 3: USER_SPACE_ATOMIC Tests
# ===========================================================================

def test_tier3_atomic_lockfile_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Simulate missing/disabled fcntl
    monkeypatch.setattr("pipeline.event_lock.fcntl", None)

    lock = WorktreeEventLock(tmp_path, timeout_seconds=2.0)
    with lock:
        assert lock.active_tier == LockTier.USER_SPACE_ATOMIC
        assert lock.diagnostic.is_fallback is True
        assert Path(lock.resolved_lock_path).exists()

        raw_meta = Path(lock.resolved_lock_path).read_text(encoding="utf-8")
        meta = json.loads(raw_meta)
        assert meta["pid"] == os.getpid()
        assert meta["lock_id"] == lock._lock_id
        assert meta["hostname"] == socket.gethostname()
        assert "acquired_at" in meta

    # Released: atomic lockfile is unlinked
    assert not Path(lock.resolved_lock_path).exists()


def test_tier3_atomic_stale_lock_dead_pid_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("pipeline.event_lock.fcntl", None)

    lock = WorktreeEventLock(tmp_path, timeout_seconds=3.0)
    atomic_path = lock._get_temp_lock_path(suffix=".atomic_lock")

    # Create stale lock belonging to non-existent PID
    dead_payload = {
        "lock_id": "dead-process-lock-uuid",
        "pid": 99999999,
        "hostname": socket.gethostname(),
        "target_path": str(tmp_path.resolve()),
        "acquired_at": time.time() - 10.0,
        "stale_timeout_seconds": 60.0,
    }
    atomic_path.write_text(json.dumps(dead_payload), encoding="utf-8")

    with lock:
        assert lock.active_tier == LockTier.USER_SPACE_ATOMIC
        assert lock.is_locked is True
        assert any("Evicted stale" in r for r in lock.diagnostic.fallback_reasons)
        # New payload written with current pid
        curr_payload = json.loads(atomic_path.read_text(encoding="utf-8"))
        assert curr_payload["pid"] == os.getpid()
        assert curr_payload["lock_id"] == lock._lock_id


def test_tier3_atomic_stale_lock_timeout_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("pipeline.event_lock.fcntl", None)

    lock = WorktreeEventLock(tmp_path, timeout_seconds=3.0, stale_timeout_seconds=1.0)
    atomic_path = lock._get_temp_lock_path(suffix=".atomic_lock")

    # Create stale lock with expired acquisition timestamp
    expired_payload = {
        "lock_id": "expired-lock-uuid",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "target_path": str(tmp_path.resolve()),
        "acquired_at": time.time() - 100.0,
        "stale_timeout_seconds": 1.0,
    }
    atomic_path.write_text(json.dumps(expired_payload), encoding="utf-8")

    with lock:
        assert lock.active_tier == LockTier.USER_SPACE_ATOMIC
        assert lock.is_locked is True
        assert any("Evicted stale" in r for r in lock.diagnostic.fallback_reasons)


def test_tier3_atomic_stale_zero_byte_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("pipeline.event_lock.fcntl", None)

    lock = WorktreeEventLock(tmp_path, timeout_seconds=3.0)
    atomic_path = lock._get_temp_lock_path(suffix=".atomic_lock")
    atomic_path.write_text("", encoding="utf-8")

    # Backdate mtime beyond grace period (10 seconds ago)
    old_time = time.time() - 10.0
    os.utime(str(atomic_path), (old_time, old_time))

    with lock:
        assert lock.active_tier == LockTier.USER_SPACE_ATOMIC
        assert lock.is_locked is True
        assert any("Evicted stale" in r for r in lock.diagnostic.fallback_reasons)


def test_tier3_release_does_not_delete_superseded_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("pipeline.event_lock.fcntl", None)

    lock = WorktreeEventLock(tmp_path, timeout_seconds=2.0)
    lock.acquire(blocking=True)
    atomic_path = Path(lock.resolved_lock_path)
    assert atomic_path.exists()

    # Simulate another worker taking over with a different lock_id
    foreign_payload = {
        "lock_id": "foreign-worker-lock-id",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "target_path": str(tmp_path.resolve()),
        "acquired_at": time.time(),
        "stale_timeout_seconds": 60.0,
    }
    atomic_path.write_text(json.dumps(foreign_payload), encoding="utf-8")

    # Original lock releases; must NOT delete foreign lock file
    lock.release()
    assert atomic_path.exists()
    atomic_path.unlink(missing_ok=True)


# ===========================================================================
# 5. Tier 4: IN_MEMORY_MUTEX Tests
# ===========================================================================

def test_tier4_in_memory_mutex_when_all_filesystem_writes_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Mock os.open to raise EROFS (Read-only filesystem) everywhere
    def mock_open(*args, **kwargs):
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(os, "open", mock_open)

    lock = WorktreeEventLock(tmp_path, timeout_seconds=2.0)
    with lock:
        assert lock.active_tier == LockTier.IN_MEMORY_MUTEX
        assert lock.diagnostic.is_fallback is True
        assert "in_memory://" in lock.resolved_lock_path
        assert any("TIER3_UNWRITABLE" in r for r in lock.diagnostic.fallback_reasons)

    assert lock.is_locked is False


def test_tier4_in_memory_mutex_thread_synchronization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def mock_open(*args, **kwargs):
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(os, "open", mock_open)

    history = []

    def task(task_id: int):
        lock = WorktreeEventLock(tmp_path, timeout_seconds=5.0, poll_interval=0.01)
        with lock:
            history.append(f"enter_{task_id}")
            time.sleep(0.08)
            history.append(f"exit_{task_id}")

    t1 = threading.Thread(target=task, args=(1,))
    t2 = threading.Thread(target=task, args=(2,))

    t1.start()
    time.sleep(0.01)
    t2.start()

    t1.join()
    t2.join()

    assert history == ["enter_1", "exit_1", "enter_2", "exit_2"]


# ===========================================================================
# 6. Async Context Manager & Concurrency Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_async_context_manager_tier1(tmp_path: Path):
    repo = tmp_path / "async_repo"
    (repo / ".git").mkdir(parents=True)

    lock = WorktreeEventLock(repo, timeout_seconds=2.0)
    async with lock:
        assert lock.is_locked is True
        assert lock.active_tier == LockTier.KERNEL_FLOCK

    assert lock.is_locked is False


@pytest.mark.asyncio
async def test_async_context_manager_tier4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def mock_open(*args, **kwargs):
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(os, "open", mock_open)

    lock = WorktreeEventLock(tmp_path, timeout_seconds=2.0)
    async with lock:
        assert lock.is_locked is True
        assert lock.active_tier == LockTier.IN_MEMORY_MUTEX

    assert lock.is_locked is False


@pytest.mark.asyncio
async def test_async_concurrent_tasks_mutual_exclusion(tmp_path: Path):
    order = []

    async def coroutine_worker(worker_id: int):
        lock = WorktreeEventLock(tmp_path, timeout_seconds=5.0, poll_interval=0.01)
        async with lock:
            order.append(f"start_{worker_id}")
            await asyncio.sleep(0.05)
            order.append(f"end_{worker_id}")

    await asyncio.gather(coroutine_worker(1), coroutine_worker(2))
    assert order in (
        ["start_1", "end_1", "start_2", "end_2"],
        ["start_2", "end_2", "start_1", "end_1"],
    )


# ===========================================================================
# 7. LockDiagnostic Data Model & Backwards Compatibility Tests
# ===========================================================================

def test_lock_diagnostic_model_and_aliases(tmp_path: Path):
    diag = LockDiagnostic(
        target_path=str(tmp_path),
        resolved_lock_path="/tmp/test.lock",
        active_tier=LockTier.TEMP_FLOCK,
        fallback_reasons=["Reason 1", "Reason 2"],
        is_locked=True,
        holder_pid=12345,
        acquired_at=time.time() - 2.0,
    )

    # Aliases
    assert diag.is_fallback is True
    assert diag.tier == LockTier.TEMP_FLOCK
    assert diag.lock_path == "/tmp/test.lock"
    assert diag.owner_pid == 12345
    assert diag.fallback_reason == "Reason 1; Reason 2"
    assert diag.held_duration_ms is not None
    assert diag.held_duration_ms > 1000.0

    # Serialization
    data = diag.to_dict()
    assert data["target_path"] == str(tmp_path)
    assert data["active_tier"] == "temp_flock"
    assert data["is_fallback"] is True
    assert data["holder_pid"] == 12345
    assert len(data["fallback_reasons"]) == 2

    # Summary string
    summary = diag.summary()
    assert "LockDiagnostic[LOCKED]" in summary
    assert "tier=temp_flock" in summary
    assert "pid=12345" in summary
