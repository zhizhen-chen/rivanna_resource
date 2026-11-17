"""Structured, HTML-free queries of the cluster's GPU resources.

Shared by the web app (app.py) and the command line entry point (rres.py), so
that both report the same numbers.  Every function returns plain dicts and
lists, ready to be printed as a table or dumped as JSON.
"""

import os
import re
import sys
import shutil
import getpass
import tempfile
import subprocess
from collections import defaultdict

from slurm_gpustat import (
    ACCEPTS_JOBS,
    DOWN,
    default_account,
    gpu_alloc_by_node,
    gpu_usage,
    my_identity,
    node_availability,
    node_tres_table,
    parse_cmd,
    parse_node_names,
    parse_size,
    parse_squeue,
)

# On-board memory per GPU in GB.  SLURM does not report it, so it is taken from,
# in this order: the file gpu_memory.conf next to this module (written by the
# user or an agent, see gpu_memory.conf.example), the type name if it carries
# the size (a100_80gb, the MIG slice 1g.24gb), or a node feature that tags the
# size (h100_94gb).  There is no built-in table: model names do not pin down
# the memory (an "a100" may be 40 or 80 GB), so unknown types show as "?".
GPU_MEMORY_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gpu_memory.conf")
_VENDOR_PREFIX = re.compile(r"^(nvidia|tesla|geforce|quadro|amd|instinct)")
_gpu_memory_table = None


def _memory_key(name: str) -> str:
    return re.sub(r"[-_ .]", "", name.lower())


def load_gpu_memory_table(path: str = GPU_MEMORY_FILE) -> dict:
    """Read the user's GPU memory table: lines "gpu_type = GB", # comments.

    Returns an empty dict if the file does not exist.  Malformed lines are
    skipped with a warning on stderr.
    """
    table = {}
    if not os.path.isfile(path):
        return table
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            name, sep, value = line.partition("=")
            try:
                if not sep or not name.strip():
                    raise ValueError
                table[_memory_key(name.strip())] = int(float(value.strip().rstrip("GgBb")))
            except ValueError:
                sys.stderr.write(f"rres: {path}:{lineno}: expected 'gpu_type = GB', "
                                 f"got {line!r}\n")
    return table


# Pending reasons that mean "waiting for resources to free up".  Other reasons
# (Dependency, QOSMaxGRESPerUser, ...) hold a job back whatever the load is,
# so they say nothing about how busy a GPU type is.
QUEUE_REASONS = {"Resources", "Priority"}

# The default wall time of a generated template, if the partition allows it.
DEFAULT_TIME = "1-00:00:00"

_GPU_REQUEST = re.compile(r"gres/gpu(?::([^:,=]+))?[:=](\d+)")


def _norm(name: str) -> str:
    return name.replace("_", "").lower()


def gpu_memory_gb(gpu_type: str, features: str = ""):
    """Return the memory of one GPU of the given type in GB, or None if unknown.

    Looked up in gpu_memory.conf first (by the type name, then without vendor
    prefixes: an entry "v100" also covers "nvidia_tesla_v100"), then read from
    the type name, then from the node's `features` (comma separated): one
    naming the GPU and its size, like "h100_94gb" for "nvidia_h100_nvl".  A
    feature counts only if its name part occurs in the type, so that a feature
    like "mem_768gb" is not taken for the GPU's memory.
    """
    global _gpu_memory_table
    if _gpu_memory_table is None:
        _gpu_memory_table = load_gpu_memory_table()
    key = _memory_key(gpu_type)
    while True:  # "nvidia_tesla_v100" -> "teslav100" -> "v100"
        if key in _gpu_memory_table:
            return _gpu_memory_table[key]
        stripped = _VENDOR_PREFIX.sub("", key)
        if stripped == key:
            break
        key = stripped
    sized = re.search(r"(\d+)gi?b", gpu_type.lower())
    if sized:
        return int(sized.group(1))
    key = _memory_key(gpu_type)
    for feature in features.split(","):
        tagged = re.fullmatch(r"([a-z0-9_]+?)_?(\d+)gi?b", feature.strip().lower())
        if tagged and tagged.group(1).replace("_", "") in key:
            return int(tagged.group(2))
    return None


def slurm_seconds(value: str) -> float:
    """Convert a SLURM duration ([D-]HH:MM:SS, MM:SS, D-HH, ...) to seconds."""
    if value.strip().lower() in ("infinite", "unlimited"):
        return float("inf")
    days, rest = 0, value.strip()
    if "-" in rest:
        days, rest = rest.split("-", 1)
        parts = [int(x) for x in rest.split(":")] + [0, 0]
        hours, minutes, seconds = parts[:3]
        days = int(days)
    else:
        parts = [int(x) for x in rest.split(":")]
        if len(parts) == 1:
            hours, minutes, seconds = 0, parts[0], 0
        elif len(parts) == 2:
            hours, (minutes, seconds) = 0, parts
        else:
            hours, minutes, seconds = parts[:3]
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _gb(value: str) -> float:
    """Parse a SLURM memory amount ("1000000M", "448G") into GB."""
    try:
        return parse_size(value, binary=True) / 1024 ** 3
    except Exception:
        return 0.0


def squeue_rows(columns, args="") -> list:
    """Run squeue (over all partitions) and split its fixed-width output.

    Args:
        columns: (field, width) pairs passed to `squeue -O`.  The widths must be
            large enough for the values, which are otherwise truncated.
        args: further squeue arguments, e.g. "-t PD" or "-u someone".

    Returns:
        one dict per job, mapping each field to its (stripped) value.
    """
    fmt = ",".join(f"{name}:{width}" for name, width in columns)
    rows = []
    for line in parse_squeue(f"-O {fmt} --noheader {args}"):
        values, offset = {}, 0
        for name, width in columns:
            values[name] = line[offset:offset + width].strip()
            offset += width
        rows.append(values)
    return rows


def partitions() -> dict:
    """Map every partition, hidden ones included, to its nodes and time limit.

    Jobs submitted to `gpu` are routed into hidden per-type partitions
    (gpu-a100-80, gpu-b200, ...), which is where pending jobs then show up.
    """
    result = {}
    for row in parse_cmd("sinfo -a -h -o '%P|%l|%N'"):
        name, limit, nodes = row.split("|")
        name = name.rstrip("*")
        part = result.setdefault(name, {"nodes": set(), "max_time": limit})
        if nodes and nodes != "(null)":
            part["nodes"].update(parse_node_names(nodes))
    return result


def gpu_nodes_with_status():
    """Return the GPU nodes worth showing, with each one's status and free GPUs.

    A node the current user cannot start a job on counts zero free GPUs and is
    shown greyed out, tagged with the reason (draining, reserved, maint), so that
    it is clear why its GPUs cannot be had.  Nodes that are down or drained are
    dropped, unless jobs still run on them, which would otherwise vanish from the
    page.

    Returns:
        (alloc_data, status, free): gpu_alloc_by_node() restricted to those nodes,
        node -> node_availability() entry, and node -> number of free GPUs.
    """
    alloc_data = gpu_alloc_by_node()
    availability = node_availability()
    status = {
        node: availability.get(
            node, {"status": DOWN, "detail": "unknown to scontrol", "state": "UNKNOWN"})
        for node in alloc_data
    }
    alloc_data = {
        node: specs
        for node, specs in alloc_data.items()
        if status[node]["status"] != DOWN or any(s["alloc"] for s in specs)
    }
    free = {
        node: sum(max(s["total"] - s["alloc"], 0) for s in specs)
        if status[node]["status"] in ACCEPTS_JOBS else 0
        for node, specs in alloc_data.items()
    }
    return alloc_data, status, free


def node_records() -> list:
    """Describe every GPU node shown on the page, one dict per node.

    `gpus_usable` is `gpus_free` unless the node has no CPU or memory left, in
    which case its idle GPUs cannot be had either.  `gpus_full_share` counts the
    free GPUs that still come with their proportional share of the node's CPUs
    and memory (total / number of GPUs), which is what the default template asks
    for: a node whose memory is used up by other jobs has idle GPUs that only a
    job asking for very little memory can get.  `planned` means the
    scheduler has set the node's idle resources aside for a pending
    higher-priority job: only jobs short enough to finish before that job starts
    can use them (backfill).
    """
    alloc_data, status, free = gpu_nodes_with_status()
    tres = node_tres_table()
    res_total = {
        node: [{"type": s["type"], "count": s["total"]} for s in specs]
        for node, specs in alloc_data.items()
    }
    users = defaultdict(dict)
    for user, by_type in gpu_usage(resources=res_total).items():
        for by_node in by_type.values():
            for node, counts in by_node.items():
                users[node][user] = users[node].get(user, 0) + counts["n_gpu"]

    records = []
    for node in sorted(alloc_data):
        spec = alloc_data[node][0]
        cfg = tres.get(node, {}).get("CfgTRES", {})
        used = tres.get(node, {}).get("AllocTRES", {})
        cpus_total = int(cfg.get("cpu", 0))
        cpus_free = max(cpus_total - int(used.get("cpu", 0)), 0)
        mem_total = _gb(cfg.get("mem", "0M"))
        mem_free = max(mem_total - _gb(used.get("mem", "0M")), 0.0)
        st = status[node]
        gpus_total = spec["total"] or 1
        full_share = min(free[node],
                         cpus_free // max(cpus_total // gpus_total, 1),
                         int(mem_free // max(mem_total / gpus_total, 1)))
        records.append({
            "node": node,
            "gpu_type": spec["type"],
            "gpu_mem_gb": gpu_memory_gb(spec["type"], spec["features"]),
            "partition": spec["partition"],
            "status": st["status"],
            "detail": st["detail"],
            "state": st.get("state", ""),
            "planned": "PLANNED" in st.get("state", "").split("+"),
            "gpus_free": free[node],
            "gpus_usable": free[node] if cpus_free > 0 and mem_free >= 1 else 0,
            "gpus_full_share": full_share,
            "gpus_total": spec["total"],
            "cpus_free": cpus_free,
            "cpus_total": cpus_total,
            "mem_free_gb": int(mem_free),
            "mem_total_gb": int(mem_total),
            "features": spec["features"],
            "users": dict(sorted(users.get(node, {}).items())),
        })
    return records


def _requested_gpus(job: dict):
    """Return (number of GPUs, requested GPU type or None) of a job's request."""
    def count(field):
        match = _GPU_REQUEST.search(job.get(field, ""))
        return (int(match.group(2)), match.group(1)) if match else (0, None)

    def as_int(field):
        value = job.get(field, "")
        return int(value) if value.isdigit() and int(value) > 0 else 1

    per_node, gtype = count("tres-per-node")
    if per_node:
        return per_node * as_int("numnodes"), gtype
    per_job, gtype = count("tres-per-job")
    if per_job:
        return per_job, gtype
    per_task, gtype = count("tres-per-task")
    if per_task:
        return per_task * as_int("numtasks"), gtype
    return 0, None


def _features_match(node_features: str, constraint: str) -> bool:
    """Approximate whether a node satisfies a job's --constraint.

    Constraints may combine features with |, &, brackets and counts; the job is
    taken to be able to run on the node if any feature it names is present.
    """
    if not constraint or constraint == "(null)":
        return True
    have = {f.strip() for f in node_features.split(",")}
    wanted = [t for t in re.split(r"[|&,()\[\]*!]", constraint) if t and not t.isdigit()]
    return any(t in have for t in wanted)


def queued_gpu_demand(records: list) -> dict:
    """Count the pending GPU jobs that wait for resources, per GPU type.

    A job counts towards every GPU type it could run on (given its partitions,
    requested GPU type and constraint), so the counts of different types may
    overlap.

    Returns:
        gpu type -> {"jobs": number of queued jobs, "gpus": GPUs they request}.
    """
    by_node = {r["node"]: r for r in records}
    parts = partitions()
    columns = [("partition", 200), ("tres-per-node", 100), ("tres-per-job", 100),
               ("tres-per-task", 100), ("numnodes", 10), ("numtasks", 10),
               ("feature", 200), ("reason", 100)]
    demand = defaultdict(lambda: {"jobs": 0, "gpus": 0})
    for job in squeue_rows(columns, "-t PD"):
        if job["reason"] not in QUEUE_REASONS:
            continue
        gpus, gtype = _requested_gpus(job)
        if not gpus:
            continue
        nodes = set()
        for part in job["partition"].split(","):
            nodes |= parts.get(part, {}).get("nodes", set())
        types = set()
        for node in nodes:
            rec = by_node.get(node)
            if rec is None:
                continue
            if gtype and not _norm(rec["gpu_type"]).startswith(_norm(gtype)):
                continue
            if not _features_match(rec["features"], job["feature"]):
                continue
            types.add(rec["gpu_type"])
        for t in types:
            demand[t]["jobs"] += 1
            demand[t]["gpus"] += gpus
    return dict(demand)


def gpu_summary(records: list = None) -> list:
    """Summarise availability and queue pressure per GPU type."""
    if records is None:
        records = node_records()
    demand = queued_gpu_demand(records)
    by_type = defaultdict(list)
    for rec in records:
        by_type[rec["gpu_type"]].append(rec)
    summary = []
    for gtype in sorted(by_type):
        recs = by_type[gtype]
        usable = [r["gpus_usable"] for r in recs]
        summary.append({
            "gpu_type": gtype,
            "gpu_mem_gb": max((r["gpu_mem_gb"] for r in recs if r["gpu_mem_gb"]),
                              default=None),
            "gpus_free": sum(usable),
            "gpus_free_full_share": sum(r["gpus_full_share"] for r in recs),
            "gpus_total": sum(r["gpus_total"] for r in recs),
            "nodes": len(recs),
            "nodes_with_free": sum(1 for u in usable if u),
            "max_free_on_one_node": max(usable, default=0),
            "planned_nodes_with_free": sum(1 for r in recs if r["gpus_usable"] and r["planned"]),
            "partitions": sorted({r["partition"] for r in recs}),
            "queued_jobs": demand.get(gtype, {}).get("jobs", 0),
            "queued_gpus": demand.get(gtype, {}).get("gpus", 0),
        })
    return summary


def _primary_partition(partition_field: str) -> str:
    names = [p for p in partition_field.split(",") if p]
    gpu_like = [p for p in names if "gpu" in p]
    return (gpu_like or names or ["gpu"])[0]


def slurm_template(gpu_type: str, alloc_data: dict = None, free: dict = None, *,
                   gpus: int = 1, nodes: int = 1, cpus: int = None, mem_gb: int = None,
                   time: str = None, account: str = None, partition: str = None,
                   node: str = None, job_name: str = "my_job",
                   command: str = "python train.py") -> str:
    """Generate an SBATCH script for the given GPU type.

    The node of that type with the most free GPUs stands for the type: its
    partition and hardware (CPUs and memory per GPU) fill in the defaults.  CPUs
    and memory default to that node's share per GPU times `gpus`, the wall time
    to DEFAULT_TIME capped at the partition's limit, and the account to the
    user's only account, if they have exactly one.

    Raises:
        ValueError: if no node fits, or the request exceeds what a node has.
    """
    if alloc_data is None:
        alloc_data, _, free = gpu_nodes_with_status()
    free = free or {}
    parts = partitions()
    candidates = [(name, specs[0]) for name, specs in alloc_data.items()
                  if specs and specs[0]["type"] == gpu_type]
    if not candidates:
        raise ValueError(f"no nodes found for GPU type {gpu_type!r}")
    if node:
        candidates = [c for c in candidates if c[0] == node]
        if not candidates:
            raise ValueError(f"{node} has no {gpu_type} GPUs")
    if partition:
        if partition not in parts:
            raise ValueError(f"unknown partition {partition!r}")
        candidates = [c for c in candidates if c[0] in parts[partition]["nodes"]]
        if not candidates:
            raise ValueError(f"partition {partition} has no {gpu_type} nodes")
    name, rep = max(candidates, key=lambda c: (free.get(c[0], 0), c[0]))

    part = partition or _primary_partition(rep.get("partition", "gpu"))
    gpu_total = rep.get("total", 1) or 1
    cpu_total = rep.get("cpu_total", 0)
    mem_total_gb = rep.get("mem_total_mb", 0) // 1024
    if not 1 <= gpus <= gpu_total:
        raise ValueError(f"{gpu_type} nodes have {gpu_total} GPUs each; "
                         f"use --nodes for more than that")
    cpus = cpus if cpus is not None else max(cpu_total // gpu_total, 1) * gpus
    mem_gb = mem_gb if mem_gb is not None else (mem_total_gb // gpu_total) * gpus
    if cpus > cpu_total:
        raise ValueError(f"{gpu_type} nodes have {cpu_total} CPUs each")
    if mem_gb > mem_total_gb:
        raise ValueError(f"{gpu_type} nodes have {mem_total_gb}G of memory each")

    max_time = parts.get(part, {}).get("max_time", "")
    limit = slurm_seconds(max_time) if max_time else float("inf")
    if time is None:
        time = DEFAULT_TIME if slurm_seconds(DEFAULT_TIME) <= limit else max_time
    elif slurm_seconds(time) > limit:
        raise ValueError(f"--time {time} exceeds the {max_time} limit of partition {part}")

    # Account: the one given, else the user's only one, else their default
    # account (what sbatch charges anyway).  Without accounting data (no
    # sacctmgr, or accounts not enforced) the line is left out and SLURM decides.
    my_accounts = sorted(my_identity()[1])
    account_note = "your Slurm account/allocation"
    if account is None:
        if len(my_accounts) == 1:
            account = my_accounts[0]
        elif my_accounts:
            account = default_account()
            others = ", ".join(my_accounts)
            if account in my_accounts:
                account_note = f"your default account; others: {others}"
            else:
                account, account_note = "<your_account>", f"one of: {others}"

    # ActiveFeatures is what SLURM enforces for --constraint; pick the most
    # specific tag (longest) that matches the gpu_type by normalised comparison
    # (ignoring underscores), so e.g. "rtxpro6000" is chosen for gpu_type
    # "rtx_pro_6000", and "a100_80gb" is chosen over "a100" for 80 GB nodes.
    # Types without such a feature (e.g. the MIG slice 1g.24gb) get no constraint.
    active = [f.strip() for f in rep.get("active_features", "").split(",") if f.strip()]
    matching = [f for f in active
                if _norm(f) == _norm(gpu_type) or _norm(gpu_type).startswith(_norm(f))]
    constraint = max(matching, key=len) if matching else None

    # Derive the base gres name from AvailableFeatures: if it contains the bare
    # GPU name without a variant suffix, use that; otherwise use gpu_type.
    # e.g. features="a100,a100_80gb,gpupod" -> gres_name="a100"
    features = [f.strip() for f in rep.get("features", "").split(",") if f.strip()]
    bare = [f for f in features
            if gpu_type.startswith(f + "_") or f == gpu_type.split("_")[0]]
    gres_name = bare[0] if bare else gpu_type
    # Nodes whose Gres has no type ("gpu:4") take an untyped request.
    gres = f"gpu:{gpus}" if gpu_type == "unknown" else f"gpu:{gres_name}:{gpus}"

    lines = ["#!/bin/bash", f"#SBATCH --partition={part}"]
    if nodes > 1:
        lines.append(f"#SBATCH --nodes={nodes}")
    if node:
        lines.append(f"#SBATCH --nodelist={node}")
    lines.append(f"#SBATCH --gres={gres}          "
                 f"# GPUs per node (max {gpu_total})")
    if constraint:
        lines.append(f"#SBATCH --constraint={constraint}")
    lines += [
        f"#SBATCH --cpus-per-task={cpus}    # CPUs (max {cpu_total} per node)",
        f"#SBATCH --mem={mem_gb}G           # memory per node (max {mem_total_gb}G)",
        f"#SBATCH --time={time}                  "
        f"# wall time (D-HH:MM:SS, max {max_time or 'unknown'} on {part})",
    ]
    if account:
        lines.append(f"#SBATCH --account={account}           # {account_note}")
    lines += [
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --output=%x-%j.out",
        "",
        "# ---- load modules / activate environment ----",
        "type module >/dev/null 2>&1 && module purge   # if the cluster uses environment modules",
        "# module load cuda/12.x",
        "",
        "# ---- your commands ----",
        command,
    ]
    return "\n".join(lines)


def estimate_start(script: str) -> dict:
    """Ask the scheduler when and where a job script would start, without submitting it.

    Wraps `sbatch --test-only`.  The estimate assumes every running job uses its
    full time limit, so jobs usually start earlier than this.

    Returns:
        {"start", "nodes", "partition", "cpus"} on success, else {"error": message}.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        proc = subprocess.run(["sbatch", "--test-only", path], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"error": "sbatch --test-only timed out"}
    finally:
        os.unlink(path)
    out = (proc.stdout + proc.stderr).strip()
    match = re.search(r"to start at (\S+).*? using (\d+) processors on nodes (\S+)"
                      r" in partition (\S+)", out)
    if proc.returncode != 0 or not match:
        return {"error": out.replace("sbatch: error: ", "") or f"sbatch exited with {proc.returncode}"}
    return {"start": match.group(1), "cpus": int(match.group(2)),
            "nodes": match.group(3), "partition": match.group(4)}


def user_jobs(user: str = None) -> list:
    """List a user's jobs (default: the current user), running and pending."""
    user = user or getpass.getuser()
    columns = [("jobid", 20), ("name", 100), ("partition", 100), ("statecompact", 6),
               ("numnodes", 10), ("numtasks", 10), ("tres-alloc", 300),
               ("tres-per-node", 100), ("tres-per-job", 100), ("tres-per-task", 100),
               ("timeused", 16), ("timelimit", 16), ("starttime", 22),
               ("nodelist", 300), ("reason", 100), ("account", 40)]
    jobs = []
    for row in squeue_rows(columns, f"-u {user}"):
        alloc = re.search(r"(?:^|,)gres/gpu=(\d+)", row["tres-alloc"])
        requested, gtype = _requested_gpus(row)
        jobs.append({
            "jobid": row["jobid"],
            "name": row["name"],
            "partition": row["partition"],
            "state": row["statecompact"],
            "gpus": int(alloc.group(1)) if alloc else requested,
            "gpu_type": gtype,
            "nodes": row["nodelist"] or None,
            "time_used": row["timeused"],
            "time_limit": row["timelimit"],
            "start_time": None if row["starttime"] in ("N/A", "") else row["starttime"],
            "reason": None if row["reason"] in ("None", "") else row["reason"],
            "account": row["account"],
        })
    return jobs


def account_balances() -> list:
    """Return the service-unit balance of each of the user's allocations.

    Balances come from `allocations`, a site tool of UVA Research Computing.  On
    other clusters the user's SLURM accounts are listed without balances.
    """
    fallback = [{"account": a, "balance": None, "reserved": None, "available": None}
                for a in sorted(my_identity()[1])]
    if not shutil.which("allocations"):
        return fallback
    try:
        raw = parse_cmd("allocations", split=False)
    except Exception:
        return fallback
    balances = []
    for line in raw.splitlines()[2:]:  # skip header and separator
        if not line.strip():
            break  # a blank line ends the account table
        tokens = line.split()
        nums = [int(t) if t.lstrip("-").isdigit() else None for t in tokens[1:4]]
        nums += [None] * (3 - len(nums))
        balances.append({"account": tokens[0], "balance": nums[0],
                         "reserved": nums[1], "available": nums[2]})
    return balances


def disk_quota() -> list:
    """Return the user's storage quotas, as reported by `hdquota`.

    `hdquota` is a site tool of UVA Research Computing; elsewhere this returns
    None, as quota tools differ too much between clusters to parse in general.
    """
    if not shutil.which("hdquota"):
        return None
    rows = []
    for line in parse_cmd("hdquota", split=False).splitlines()[2:]:
        tokens = line.split()
        if len(tokens) < 9:
            continue
        # "<type, possibly several words> <location> <n> <unit> <n> <unit> <n> <unit> <pct>"
        loc, size, used, avail, pct = (tokens[-8], " ".join(tokens[-7:-5]),
                                       " ".join(tokens[-5:-3]), " ".join(tokens[-3:-1]),
                                       tokens[-1])
        rows.append({"storage": " ".join(tokens[:-8]), "location": loc, "size": size,
                     "used": used, "available": avail, "use_percent": int(pct.rstrip("%"))})
    return rows
