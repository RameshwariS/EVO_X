"""Single-writer file registry with an authoritative append-only event journal.

Model artifacts are atomically written before their registration event. Each
journal append is flushed/fsynced; replay reconstructs the active pointer and
version states. No mutable manifest can disagree with the history after a crash.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from evox.models.ids import IDSModel


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class RegistryError(ValueError):
    pass


class FileModelRegistry:
    """One owner per registry directory; independent experiments use separate roots.

    An exclusive lifetime lease rejects a second process/instance rather than
    allowing competing active-pointer owners. Close the registry to release it.
    """
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = self.root / "models"
        self.artifacts.mkdir(exist_ok=True)
        self.journal = self.root / "history.jsonl"
        self._mutex = threading.RLock()
        self._lease = (self.root / ".owner.lock").open("a+b")
        self._lease.seek(0, os.SEEK_END)
        if self._lease.tell() == 0:
            self._lease.write(b"0"); self._lease.flush()
        self._lease.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lease.close()
            raise RegistryError("Registry already has an active owner") from exc
        self.versions, self.events, self.active_version = {}, [], None
        try:
            self._replay()
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self._lease.closed:
            # Closing the handle releases the OS lease on Windows and Unix.
            self._lease.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _replay(self):
        if not self.journal.exists():
            return
        previous = "0" * 64
        with self.journal.open("r", encoding="utf-8") as f:
            for sequence, line in enumerate(f, 1):
                if not line.endswith("\n"):
                    raise RegistryError("Incomplete journal tail; preserve the file for recovery")
                try:
                    event = json.loads(line)
                    digest = event.pop("hash")
                    if event["sequence"] != sequence or event["previous_hash"] != previous:
                        raise RegistryError("Invalid registry event sequence/hash chain")
                    if hashlib.sha256(canonical(event).encode()).hexdigest() != digest:
                        raise RegistryError("Registry journal checksum mismatch")
                    event["hash"] = digest
                    if event["active_before"] != self.active_version:
                        raise RegistryError("Registry active-pointer chain is inconsistent")
                    self._apply(event)
                    previous = digest
                except (KeyError, json.JSONDecodeError) as exc:
                    raise RegistryError("Malformed registry journal") from exc

    def _apply(self, event):
        for record in event["updates"]:
            self.versions[record["version"]] = record
        self.active_version = event["active_after"]
        self.events.append(event)

    def _append(self, kind, payload=None, updates=(), active_after=None):
        if self._lease.closed:
            raise RegistryError("Registry is closed")
        event = dict(schema=1, sequence=len(self.events) + 1,
                     timestamp=datetime.now(timezone.utc).isoformat(), kind=kind, payload=payload or {},
                     active_before=self.active_version,
                     active_after=active_after if active_after is not None else self.active_version,
                     updates=list(updates), previous_hash=self.events[-1]["hash"] if self.events else "0" * 64)
        # Journal snapshots must not change when a caller later mutates context
        # or monitoring evidence dictionaries.
        event = json.loads(canonical(event))
        event["hash"] = hashlib.sha256(canonical(event).encode()).hexdigest()
        with self.journal.open("a", encoding="utf-8", newline="\n") as f:
            f.write(canonical(event) + "\n"); f.flush(); os.fsync(f.fileno())
        self._apply(event)
        return event

    def record_event(self, kind, payload):
        with self._mutex:
            return self._append(kind, payload)

    def _save(self, model, version):
        target = self.artifacts / f"{version}.joblib"
        temporary = self.artifacts / f".{version}-{uuid.uuid4().hex}.joblib"
        model.save(temporary)
        with temporary.open("r+b") as f:
            os.fsync(f.fileno())
        os.replace(temporary, target)
        return dict(artifact=f"models/{version}.joblib", sha256=hashlib.sha256(target.read_bytes()).hexdigest())

    def initialize(self, model, training_ids=(), metadata=None, training_snapshot_id=None):
        with self._mutex:
            if self.versions:
                raise RegistryError("Registry is already initialized")
            record = dict(version="v1", parent_version=None, candidate_id="initial", status="DEPLOYED",
                          training_ids=list(training_ids), metadata=metadata or {}, action="initial",
                          training_snapshot_id=training_snapshot_id or "ids:" + hashlib.sha256(canonical(list(training_ids)).encode()).hexdigest(),
                          model_kind=model.kind, model_params=model.params, model_seed=model.seed,
                          **self._save(model, "v1"))
            self._append("INITIALIZED", updates=[record], active_after="v1")
            return "v1"

    def candidate_version(self, candidate_id):
        return next((v for v, r in self.versions.items() if r["candidate_id"] == candidate_id), None)

    def register(self, candidate, context=None):
        with self._mutex:
            if candidate.status == "NO_OP":
                raise RegistryError("No-Op is an event, not a new model version")
            existing = self.candidate_version(candidate.candidate_id)
            if existing:
                record = self.versions[existing]
                if (record["parent_version"] != candidate.parent_version or
                        record["training_snapshot_id"] != candidate.training_snapshot_id):
                    raise RegistryError("Candidate ID collision")
                return existing
            if candidate.parent_version != self.active_version:
                raise RegistryError("Stale candidate parent")
            version = f"v{len(self.versions) + 1}"
            record = dict(version=version, **candidate.record(), context=context or {},
                          **self._save(candidate.model, version))
            self._append("CANDIDATE_REGISTERED", updates=[record])
            return version

    def reject(self, version, evaluation, decision):
        with self._mutex:
            record = self.versions[version]
            if record["status"] != "CANDIDATE":
                raise RegistryError("Only a pending candidate can be rejected")
            record = dict(record, status="REJECTED", metrics=evaluation, gate=decision)
            self._append("REJECTED", updates=[record], payload=dict(version=version))

    def promote(self, version, evaluation, decision):
        with self._mutex:
            record = self.versions[version]
            if record["status"] != "CANDIDATE" or record["parent_version"] != self.active_version:
                raise RegistryError("Only a pending candidate of the active model can be promoted")
            if not decision.get("promote"):
                raise RegistryError("Promotion requires an accepting policy decision")
            previous = dict(self.versions[self.active_version], status="SUPERSEDED")
            record = dict(record, status="DEPLOYED", metrics=evaluation, gate=decision)
            self._append("PROMOTED", updates=[previous, record], active_after=version,
                         payload=dict(version=version, parent_version=previous["version"]))

    def rollback(self, reason, evidence=None):
        with self._mutex:
            current = self.versions[self.active_version]
            target = current["parent_version"]
            if target is None or self.versions[target]["status"] not in ("SUPERSEDED", "DEPLOYED"):
                raise RegistryError("No known-good parent is available for rollback")
            failed = dict(current, status="ROLLED_BACK")
            restored = dict(self.versions[target], status="DEPLOYED")
            self._append("ROLLED_BACK", updates=[failed, restored], active_after=target,
                         payload=dict(failed_version=current["version"], restored_version=target,
                                      reason=reason, evidence=evidence or {}))
            return target

    def load(self, version=None):
        record = self.versions[version or self.active_version]
        path = self.root / record["artifact"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RegistryError("Model artifact checksum mismatch or missing artifact")
        return IDSModel.load(path)

    def ancestor_training_ids(self, version=None):
        ids, current = set(), version or self.active_version
        while current is not None:
            record = self.versions[current]
            ids.update(record["training_ids"])
            current = record["parent_version"]
        return ids
