import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class Job:
    def __init__(self, kind, params, directory: Path):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.params, self.dir = kind, params, directory / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status = "queued"   # queued | processing | completed | failed
        self.error = None
        self.created = time.time()
        self.src: Path | None = None
        self.output: Path | None = None
        self.csv: Path | None = None
        self.collector = None
        self.processed_frames = 0
        self.total_frames = 0
        self.duration_s = 0.0
        self.video_info: dict = {}
        self.results: list = []
        self.cancel = threading.Event()

    @property
    def progress(self):
        if self.status == "completed":
            return 1.0
        return min(self.processed_frames / self.total_frames, 1.0) if self.total_frames else 0.0

    def to_dict(self):
        d = {"id": self.id, "kind": self.kind, "status": self.status, "error": self.error,
             "progress": round(self.progress, 4), "processed_frames": self.processed_frames,
             "total_frames": self.total_frames, "video_duration_s": round(self.duration_s, 2),
             "video_info": self.video_info, "params": self.params, "results": self.results,
             "has_output": bool(self.output and self.output.exists()),
             "stats": self.collector.summary() if self.collector else None}
        return d


class JobManager:
    def __init__(self, base: Path, ttl: int, max_jobs: int):
        self.base, self.ttl, self.max_jobs = Path(base), ttl, max_jobs
        self.base.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vt-job")
        self.latest_benchmark: list = []
        self.cleanup(all_orphans=True)

    def new(self, kind, params) -> Job:
        job = Job(kind, params, self.base)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id) -> Job | None:
        return self._jobs.get(job_id)

    def discard(self, job_id):
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job:
            shutil.rmtree(job.dir, ignore_errors=True)

    def submit(self, job: Job, fn, *args):
        def runner():
            try:
                fn(job, *args)
            except Exception as exc:  # last-resort guard; job fns handle their own errors
                job.status, job.error = "failed", f"Unexpected error: {exc}"
        self._pool.submit(runner)

    def cleanup(self, all_orphans=False):
        """Delete expired jobs, and the oldest ones beyond the cap. Active jobs are kept."""
        now = time.time()
        with self._lock:
            finished = sorted((j for j in self._jobs.values() if j.status in ("completed", "failed")),
                              key=lambda j: j.created)
            drop = [j for j in finished if now - j.created > self.ttl]
            extra = len(self._jobs) - len(drop) - self.max_jobs
            drop += [j for j in finished if j not in drop][:max(extra, 0)]
            for j in drop:
                self._jobs.pop(j.id, None)
                shutil.rmtree(j.dir, ignore_errors=True)
            if all_orphans:  # leftovers from a previous process
                known = {j.id for j in self._jobs.values()}
                for p in self.base.iterdir():
                    if p.is_dir() and p.name not in known:
                        shutil.rmtree(p, ignore_errors=True)
