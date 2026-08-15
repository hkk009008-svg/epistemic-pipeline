"""Robust multi-tier worktree coordination and file locking for Epistemic Pipeline.

Supports:
- Standard Git repositories and Git linked worktrees (.git pointer files)
- Read-only sandbox mounts (EROFS / EACCES)
- 4-Tier fallback hierarchy:
    Tier 1: Kernel flock on resolved .git directory (worktree_event.lock)
    Tier 2: Tempfile flock on /tmp/epistemic_worktree_<slug>_<sha256[:16]>.lock
    Tier 3: User-space atomic O_CREAT | O_EXCL lockfile with JSON metadata & stale lock recovery
    Tier 4: In-process threading.RLock and asyncio.Lock mutex fallback
- Race-free atomic quarantine stale lock recovery (os.rename)
- Dual synchronous (with) and asynchronous (async with) context managers
- Full diagnostic inspection (LockDiagnostic)
"""
from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import logging
import os
import re
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Self

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global In-Process Synchronization Registries (Tier 4)
# ---------------------------------------------------------------------------

_IN_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_IN_PROCESS_REGISTRY_LOCK = threading.Lock()

_ASYNC_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}
_ASYNC_REGISTRY_LOCK = threading.Lock()


def _get_in_process_thread_lock(canonical_key: str) -> threading.RLock:
    with _IN_PROCESS_REGISTRY_LOCK:
        if canonical_key not in _IN_PROCESS_LOCKS:
            _IN_PROCESS_LOCKS[canonical_key] = threading.RLock()
        return _IN_PROCESS_LOCKS[canonical_key]


def _get_in_process_async_lock(canonical_key: str) -> asyncio.Lock:
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        loop_id = 0

    key = (loop_id, canonical_key)
    with _ASYNC_REGISTRY_LOCK:
        if key not in _ASYNC_LOCKS:
            _ASYNC_LOCKS[key] = asyncio.Lock()
        return _ASYNC_LOCKS[key]


# ---------------------------------------------------------------------------
# Data Models & Enums
# ---------------------------------------------------------------------------

class LockTier(str, Enum):
    KERNEL_FLOCK = "kernel_flock"
    TEMP_FLOCK = "temp_flock"
    USER_SPACE_ATOMIC = "user_space_atomic"
    IN_MEMORY_MUTEX = "in_memory_mutex"


@dataclass
class LockDiagnostic:
    target_path: str
    resolved_lock_path: str | None = None
    active_tier: LockTier = LockTier.KERNEL_FLOCK
    fallback_reasons: list[str] = field(default_factory=list)
    is_locked: bool = False
    holder_pid: int | None = None
    acquired_at: float | None = None
    acquisition_duration_ms: float | None = None
    _held_duration_ms: float | None = field(default=None, repr=False)
    lock_id: str | None = None

    def __call__(self) -> LockDiagnostic:
        """Enables both lock.diagnostic() method call and lock.diagnostic property access."""
        return self

    @property
    def held_duration_ms(self) -> float | None:
        """Current duration the lock has been held in milliseconds, if locked."""
        if self._held_duration_ms is not None:
            return self._held_duration_ms
        if self.is_locked and self.acquired_at is not None:
            return (time.time() - self.acquired_at) * 1000.0
        return None

    @held_duration_ms.setter
    def held_duration_ms(self, val: float | None) -> None:
        self._held_duration_ms = val

    @property
    def is_fallback(self) -> bool:
        """True if lock acquisition fell back from Tier 1 (KERNEL_FLOCK)."""
        return self.active_tier != LockTier.KERNEL_FLOCK

    @property
    def tier(self) -> LockTier:
        """Backwards-compatible alias for active_tier."""
        return self.active_tier

    @property
    def lock_path(self) -> str | None:
        """Backwards-compatible alias for resolved_lock_path."""
        return self.resolved_lock_path

    @property
    def owner_pid(self) -> int | None:
        """Backwards-compatible alias for holder_pid."""
        return self.holder_pid

    @property
    def fallback_reason(self) -> str | None:
        """Backwards-compatible single fallback reason string (or None)."""
        if not self.fallback_reasons:
            return None
        return "; ".join(self.fallback_reasons)

    def to_dict(self) -> dict[str, Any]:
        """Serializes diagnostic metadata into a dictionary."""
        return {
            "target_path": self.target_path,
            "resolved_lock_path": self.resolved_lock_path,
            "active_tier": self.active_tier.value,
            "fallback_reasons": list(self.fallback_reasons),
            "is_locked": self.is_locked,
            "is_fallback": self.is_fallback,
            "holder_pid": self.holder_pid,
            "acquired_at": self.acquired_at,
            "acquisition_duration_ms": self.acquisition_duration_ms,
            "held_duration_ms": self.held_duration_ms,
            "lock_id": self.lock_id,
        }

    def summary(self) -> str:
        """Single-line human-readable diagnostic summary."""
        status = "LOCKED" if self.is_locked else "UNLOCKED"
        fallback_tag = f" (fallback: {len(self.fallback_reasons)} hops)" if self.is_fallback else ""
        return (
            f"LockDiagnostic[{status}] tier={self.active_tier.value}{fallback_tag} "
            f"pid={self.holder_pid} path={self.resolved_lock_path}"
        )


# ---------------------------------------------------------------------------
# Git Directory Resolution Helper
# ---------------------------------------------------------------------------

def resolve_git_dir(target_path: str | Path) -> Path:
    """Resolves the true git administrative directory for repositories and worktrees.

    Handles:
    - Standard repositories (.git is a directory)
    - Linked worktrees (.git is a file containing 'gitdir: <path>')
    - Subdirectories within repositories (ascends parent directories)
    - Relative and absolute gitdir paths
    - Bare git repositories (HEAD, objects/, refs/ in directory)
    - Non-git directories (gracefully falls back to input path without raising NotADirectoryError)
    - Corrupted or unreadable .git pointer files

    Returns:
        Path: A resolved directory path where git metadata or fallback locks can reside.
    """
    target = Path(target_path).expanduser().resolve()

    if target.name == ".git":
        if target.is_dir():
            return target
        if target.is_file():
            return _parse_gitdir_pointer(target, target.parent)

    if _is_bare_git_repo(target):
        return target

    curr = target
    while True:
        git_entry = curr / ".git"
        try:
            if git_entry.is_dir():
                return git_entry.resolve()
            if git_entry.is_file():
                return _parse_gitdir_pointer(git_entry, curr)
        except (PermissionError, OSError):
            pass

        if _is_bare_git_repo(curr):
            return curr.resolve()

        parent = curr.parent
        if parent == curr:
            break
        curr = parent

    return target


def _parse_gitdir_pointer(git_file: Path, base_dir: Path) -> Path:
    """Safely extracts and resolves the 'gitdir: <path>' from a .git pointer file."""
    try:
        content = git_file.read_text(encoding="utf-8", errors="replace").strip()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("gitdir:"):
                raw_path = line[len("gitdir:"):].strip()
                if not raw_path:
                    continue
                path_obj = Path(raw_path)
                if not path_obj.is_absolute():
                    return (base_dir / path_obj).resolve()
                return path_obj.resolve()
    except (OSError, UnicodeDecodeError, PermissionError):
        pass
    return base_dir.resolve()


def _is_bare_git_repo(path: Path) -> bool:
    """Checks if a directory is a bare git repository."""
    try:
        return (
            path.is_dir()
            and (path / "HEAD").is_file()
            and (path / "objects").is_dir()
            and (path / "refs").is_dir()
        )
    except (OSError, PermissionError):
        return False


# ---------------------------------------------------------------------------
# WorktreeEventLock Implementation
# ---------------------------------------------------------------------------

class WorktreeEventLock:
    """Multi-tier fallback lock coordinator for Git worktrees and sandboxes."""

    def __init__(
        self,
        target_path: str | Path,
        timeout_seconds: float = 30.0,
        stale_timeout_seconds: float = 60.0,
        poll_interval: float = 0.05,
        lock_filename: str = "worktree_event.lock",
        timeout: float | None = None,
    ):
        self.target_path = str(Path(target_path).resolve())
        self.timeout_seconds = float(timeout if timeout is not None else timeout_seconds)
        self.stale_timeout_seconds = float(stale_timeout_seconds)
        self.poll_interval = float(poll_interval)
        self.lock_filename = str(lock_filename)
        self._lock_id = uuid.uuid4().hex

        self.diagnostic = LockDiagnostic(
            target_path=self.target_path,
            resolved_lock_path=None,
            active_tier=LockTier.KERNEL_FLOCK,
            fallback_reasons=[],
            is_locked=False,
            lock_id=self._lock_id,
        )

        self._is_locked = False
        self._fd: int | None = None
        self._lock_path: Path | None = None
        self._active_tier: LockTier | None = None
        self._in_memory_lock: threading.RLock | None = None
        self._async_lock: asyncio.Lock | None = None

    @property
    def is_locked(self) -> bool:
        return self._is_locked

    @property
    def active_tier(self) -> LockTier | None:
        return self._active_tier

    @property
    def resolved_lock_path(self) -> str | None:
        return self.diagnostic.resolved_lock_path

    # -----------------------------------------------------------------------
    # Synchronous Acquisition & Release
    # -----------------------------------------------------------------------

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Acquires lock adhering to 4-tier fallback hierarchy."""
        effective_timeout = float(timeout if timeout is not None else self.timeout_seconds)
        start_time = time.time()
        deadline = start_time + (effective_timeout if blocking else 0.0)

        while True:
            if self._try_acquire_sync():
                now = time.time()
                self._is_locked = True
                self.diagnostic.is_locked = True
                self.diagnostic.acquired_at = now
                self.diagnostic.acquisition_duration_ms = (now - start_time) * 1000.0
                self.diagnostic.holder_pid = os.getpid()
                self.diagnostic.lock_id = self._lock_id
                return True

            if not blocking or time.time() >= deadline:
                return False

            time.sleep(self.poll_interval)

    def release(self) -> None:
        """Releases the acquired lock according to active tier semantics."""
        if not self._is_locked:
            return

        now = time.time()
        if self.diagnostic.acquired_at is not None:
            self.diagnostic.held_duration_ms = (now - self.diagnostic.acquired_at) * 1000.0

        try:
            if self._active_tier in (LockTier.KERNEL_FLOCK, LockTier.TEMP_FLOCK):
                self._release_flock()
            elif self._active_tier == LockTier.USER_SPACE_ATOMIC:
                self._release_atomic()
            elif self._active_tier == LockTier.IN_MEMORY_MUTEX:
                self._release_in_memory()
        finally:
            self._is_locked = False
            self.diagnostic.is_locked = False

    def get_diagnostic(self) -> LockDiagnostic:
        """Returns structured diagnostic inspection."""
        return self.diagnostic

    def __enter__(self) -> Self:
        if not self.acquire(blocking=True):
            raise TimeoutError(
                f"Failed to acquire WorktreeEventLock for {self.target_path} "
                f"within {self.timeout_seconds}s (diagnostic: {self.diagnostic.to_dict()})"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # -----------------------------------------------------------------------
    # Asynchronous Acquisition & Release
    # -----------------------------------------------------------------------

    async def acquire_async(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Asynchronously acquires lock adhering to 4-tier fallback hierarchy."""
        effective_timeout = float(timeout if timeout is not None else self.timeout_seconds)
        start_time = time.time()
        deadline = start_time + (effective_timeout if blocking else 0.0)

        while True:
            if await self._try_acquire_async():
                now = time.time()
                self._is_locked = True
                self.diagnostic.is_locked = True
                self.diagnostic.acquired_at = now
                self.diagnostic.acquisition_duration_ms = (now - start_time) * 1000.0
                self.diagnostic.holder_pid = os.getpid()
                self.diagnostic.lock_id = self._lock_id
                return True

            if not blocking or time.time() >= deadline:
                return False

            await asyncio.sleep(self.poll_interval)

    async def release_async(self) -> None:
        """Asynchronously releases the acquired lock."""
        self.release()

    async def __aenter__(self) -> Self:
        if not await self.acquire_async(blocking=True):
            raise TimeoutError(
                f"Failed to acquire async WorktreeEventLock for {self.target_path} "
                f"within {self.timeout_seconds}s (diagnostic: {self.diagnostic.to_dict()})"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.release_async()

    # -----------------------------------------------------------------------
    # Multi-Tier Sync / Async Dispatch
    # -----------------------------------------------------------------------

    def _try_acquire_sync(self) -> bool:
        t1_ok, t1_contention = self._try_tier1_flock()
        if t1_ok:
            return True
        if t1_contention:
            return False

        t2_ok, t2_contention = self._try_tier2_temp_flock()
        if t2_ok:
            return True
        if t2_contention:
            return False

        t3_ok, t3_contention = self._try_tier3_atomic()
        if t3_ok:
            return True
        if t3_contention:
            return False

        return self._try_tier4_thread_mutex()

    async def _try_acquire_async(self) -> bool:
        t1_ok, t1_contention = self._try_tier1_flock()
        if t1_ok:
            return True
        if t1_contention:
            return False

        t2_ok, t2_contention = self._try_tier2_temp_flock()
        if t2_ok:
            return True
        if t2_contention:
            return False

        t3_ok, t3_contention = self._try_tier3_atomic()
        if t3_ok:
            return True
        if t3_contention:
            return False

        return await self._try_tier4_async_mutex()

    # -----------------------------------------------------------------------
    # Tier 1: Kernel flock on Git Directory
    # -----------------------------------------------------------------------

    def _try_tier1_flock(self) -> tuple[bool, bool]:
        if fcntl is None:
            self._record_fallback("FCNTL_UNAVAILABLE", LockTier.TEMP_FLOCK)
            return False, False

        try:
            git_dir = resolve_git_dir(self.target_path)
            if not git_dir.is_dir():
                try:
                    git_dir.mkdir(parents=True, exist_ok=True)
                except (PermissionError, OSError) as exc:
                    self._record_fallback(f"GIT_DIR_UNWRITABLE ({exc})", LockTier.TEMP_FLOCK)
                    return False, False

            lock_file = git_dir / self.lock_filename
            self.diagnostic.resolved_lock_path = str(lock_file)
            fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                self._lock_path = lock_file
                self._active_tier = LockTier.KERNEL_FLOCK
                self.diagnostic.active_tier = LockTier.KERNEL_FLOCK
                return True, False
            except (BlockingIOError, OSError) as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                    os.close(fd)
                    return False, True  # Valid contention on Tier 1
                os.close(fd)
                self._record_fallback(f"TIER1_FLOCK_FAILED (errno {exc.errno}: {exc})", LockTier.TEMP_FLOCK)
                return False, False
        except (PermissionError, OSError) as exc:
            self._record_fallback(f"TIER1_PATH_ERROR ({exc})", LockTier.TEMP_FLOCK)
            return False, False

    # -----------------------------------------------------------------------
    # Tier 2: Temp Directory flock
    # -----------------------------------------------------------------------

    def _try_tier2_temp_flock(self) -> tuple[bool, bool]:
        if fcntl is None:
            self._record_fallback("FCNTL_UNAVAILABLE", LockTier.USER_SPACE_ATOMIC)
            return False, False

        try:
            temp_lock_file = self._get_temp_lock_path(suffix=".lock")
            self.diagnostic.resolved_lock_path = str(temp_lock_file)
            fd = os.open(str(temp_lock_file), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                self._lock_path = temp_lock_file
                self._active_tier = LockTier.TEMP_FLOCK
                self.diagnostic.active_tier = LockTier.TEMP_FLOCK
                return True, False
            except (BlockingIOError, OSError) as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                    os.close(fd)
                    return False, True  # Valid contention on Tier 2
                os.close(fd)
                self._record_fallback(f"TIER2_FLOCK_FAILED (errno {exc.errno}: {exc})", LockTier.USER_SPACE_ATOMIC)
                return False, False
        except (PermissionError, OSError) as exc:
            self._record_fallback(f"TIER2_TEMP_UNWRITABLE ({exc})", LockTier.USER_SPACE_ATOMIC)
            return False, False

    # -----------------------------------------------------------------------
    # Tier 3: User-Space Atomic O_CREAT | O_EXCL Lockfile
    # -----------------------------------------------------------------------

    def _try_tier3_atomic(self) -> tuple[bool, bool]:
        try:
            atomic_lock_file = self._get_temp_lock_path(suffix=".atomic_lock")
            self.diagnostic.resolved_lock_path = str(atomic_lock_file)

            try:
                flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
                fd = os.open(str(atomic_lock_file), flags, 0o600)
                payload = {
                    "lock_id": self._lock_id,
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "target_path": self.target_path,
                    "acquired_at": time.time(),
                    "stale_timeout_seconds": self.stale_timeout_seconds,
                }
                payload_bytes = json.dumps(payload, indent=2).encode("utf-8")
                os.write(fd, payload_bytes)
                os.fsync(fd)
                os.close(fd)

                self._lock_path = atomic_lock_file
                self._active_tier = LockTier.USER_SPACE_ATOMIC
                self.diagnostic.active_tier = LockTier.USER_SPACE_ATOMIC
                return True, False

            except FileExistsError:
                if self._check_and_evict_stale_tier3(atomic_lock_file):
                    return False, True  # Stale lock evicted; retryable contention
                return False, True      # Active lock contention

        except (PermissionError, OSError) as exc:
            self._record_fallback(f"TIER3_UNWRITABLE ({exc})", LockTier.IN_MEMORY_MUTEX)
            return False, False

    def _check_and_evict_stale_tier3(self, lock_path: Path) -> bool:
        """Inspects lockfile metadata and evicts if confirmed stale via atomic rename."""
        metadata = None
        is_stale = False

        try:
            raw = lock_path.read_text(encoding="utf-8")
            if not raw.strip():
                mtime = lock_path.stat().st_mtime
                if (time.time() - mtime) > 5.0:
                    is_stale = True
            else:
                metadata = json.loads(raw)
                pid = metadata.get("pid")
                hostname = metadata.get("hostname")
                acquired_at = metadata.get("acquired_at", 0.0)
                stale_timeout = metadata.get("stale_timeout_seconds", self.stale_timeout_seconds)

                if hostname == socket.gethostname() and isinstance(pid, int) and not self._is_pid_alive(pid):
                    is_stale = True

                if (time.time() - acquired_at) > stale_timeout:
                    is_stale = True
        except (FileNotFoundError, OSError):
            return False
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            try:
                if (time.time() - lock_path.stat().st_mtime) > 5.0:
                    is_stale = True
            except OSError:
                return False

        if not is_stale:
            return False

        quarantine_name = (
            f"{lock_path.name}.stale.{os.getpid()}."
            f"{time.time_ns()}.{uuid.uuid4().hex[:8]}"
        )
        quarantine_path = lock_path.parent / quarantine_name

        try:
            os.rename(str(lock_path), str(quarantine_path))
        except (FileNotFoundError, OSError):
            return False

        try:
            os.unlink(str(quarantine_path))
        except OSError:
            pass

        holder_info = f"PID {metadata.get('pid')}" if metadata else "unreadable"
        reason = f"Evicted stale Tier 3 lock ({holder_info} on {lock_path})"
        if reason not in self.diagnostic.fallback_reasons:
            self.diagnostic.fallback_reasons.append(reason)
        return True

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    # -----------------------------------------------------------------------
    # Tier 4: In-Memory Mutex Fallback
    # -----------------------------------------------------------------------

    def _try_tier4_thread_mutex(self) -> bool:
        canonical_key = self.target_path
        self._active_tier = LockTier.IN_MEMORY_MUTEX
        self.diagnostic.active_tier = LockTier.IN_MEMORY_MUTEX
        self.diagnostic.resolved_lock_path = f"in_memory://{canonical_key}"

        thread_lock = _get_in_process_thread_lock(canonical_key)
        acquired = thread_lock.acquire(blocking=False)
        if acquired:
            self._in_memory_lock = thread_lock
            return True
        return False

    async def _try_tier4_async_mutex(self) -> bool:
        canonical_key = self.target_path
        self._active_tier = LockTier.IN_MEMORY_MUTEX
        self.diagnostic.active_tier = LockTier.IN_MEMORY_MUTEX
        self.diagnostic.resolved_lock_path = f"in_memory://{canonical_key}"

        async_lock = _get_in_process_async_lock(canonical_key)
        if not async_lock.locked():
            await async_lock.acquire()
            self._async_lock = async_lock
            return True
        return False

    # -----------------------------------------------------------------------
    # Release Operations
    # -----------------------------------------------------------------------

    def _release_flock(self) -> None:
        if self._fd is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def _release_atomic(self) -> None:
        if self._lock_path is not None and self._lock_path.exists():
            try:
                raw = self._lock_path.read_text(encoding="utf-8")
                current_meta = json.loads(raw)
                if current_meta.get("lock_id") == self._lock_id:
                    self._lock_path.unlink(missing_ok=True)
                else:
                    logger.warning(
                        "Tier 3 lock for %s was superseded by another process. Skipping deletion.",
                        self.target_path,
                    )
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError, OSError):
                try:
                    self._lock_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _release_in_memory(self) -> None:
        if self._in_memory_lock is not None:
            try:
                self._in_memory_lock.release()
            except RuntimeError:
                pass
            self._in_memory_lock = None

        if self._async_lock is not None:
            try:
                self._async_lock.release()
            except RuntimeError:
                pass
            self._async_lock = None

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _get_temp_lock_path(self, suffix: str = ".lock") -> Path:
        target_hash = hashlib.sha256(self.target_path.encode("utf-8")).hexdigest()[:16]
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(self.target_path).name)[:24]
        filename = f"epistemic_worktree_{safe_name}_{target_hash}{suffix}"
        return Path(tempfile.gettempdir()) / filename

    def _record_fallback(self, reason: str, next_tier: LockTier) -> None:
        if reason not in self.diagnostic.fallback_reasons:
            self.diagnostic.fallback_reasons.append(reason)
            logger.debug(
                "WorktreeEventLock for %s: %s. Falling back towards %s",
                self.target_path,
                reason,
                next_tier.value,
            )
