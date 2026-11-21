#!/usr/bin/env python
"""rres: query the GPU resources of a Slurm cluster from the command line.

A thin wrapper over sinfo / squeue / scontrol / sbatch --test-only, meant for
both people and agents.  It needs only python3 and the Slurm client commands;
`accounts` and `quota` show more on clusters with UVA's `allocations` and
`hdquota` tools.  Every subcommand takes --json for machine-readable
output.  Typical workflow:

    rres gpus                          # free GPUs and queue length per GPU type
    rres eta --gpus 2 --time 6:00:00   # which GPU type would start such a job first
    rres template a100_80gb --gpus 2 -o job.sh && sbatch job.sh

Nothing here submits or cancels jobs: `eta` uses `sbatch --test-only`.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from datetime import datetime

# Run from anywhere: the modules live next to this file.
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import resource_query as rq  # noqa: E402


def _table(rows, columns):
    """Print rows (dicts) as an aligned table; columns are (header, key or callable)."""
    cells = [[h for h, _ in columns]]
    for row in rows:
        cells.append([str(get(row) if callable(get) else row.get(get, "")) for _, get in columns])
    widths = [max(len(r[i]) for r in cells) for i in range(len(columns))]
    for r in cells:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())


def _emit(args, data, printer):
    if args.json:
        json.dump(data, sys.stdout, indent=2, default=list)
        print()
    else:
        printer(data)


def _wait_minutes(start: str):
    try:
        # strptime rather than fromisoformat: rres also runs on python 3.6.
        begin = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
        return max(int((begin - datetime.now()).total_seconds() // 60), 0)
    except (TypeError, ValueError):
        return None


def _fmt_wait(minutes):
    if minutes is None:
        return "?"
    if minutes < 60:
        return f"{minutes}m"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f}h"
    return f"{minutes / 1440:.1f}d"


# --------------------------------------------------------------------- commands

def cmd_gpus(args):
    summary = rq.gpu_summary()
    if args.type:
        summary = [s for s in summary if s["gpu_type"] == args.type]
    if args.min_mem:
        summary = [s for s in summary if (s["gpu_mem_gb"] or 0) >= args.min_mem]

    def show(rows):
        _table(rows, [
            ("TYPE", "gpu_type"),
            ("MEM", lambda r: f"{r['gpu_mem_gb']}G" if r["gpu_mem_gb"] else "?"),
            ("FREE/TOTAL", lambda r: f"{r['gpus_free']}/{r['gpus_total']}"),
            ("FULL_SHARE", "gpus_free_full_share"),
            ("NODES_W_FREE", lambda r: f"{r['nodes_with_free']}/{r['nodes']}"),
            ("MAX_FREE_1NODE", "max_free_on_one_node"),
            ("QUEUED_JOBS", "queued_jobs"),
            ("QUEUED_GPUS", "queued_gpus"),
            ("PARTITION", lambda r: ",".join(r["partitions"])),
        ])
        print("\nFREE counts GPUs on nodes that accept your jobs and still have CPU and "
              "memory left;\nFULL_SHARE those that still come with their share of the "
              "node's CPUs and memory\n(`rres template` asks for the CPU share by default).\nQUEUED = pending GPU jobs waiting for resources that could run "
              "on the type.\nFree GPUs do not mean an immediate start: run `rres eta` for "
              "the scheduler's estimate.")
    _emit(args, summary, show)


def cmd_nodes(args):
    records = rq.node_records()
    if args.type:
        records = [r for r in records if r["gpu_type"] == args.type]
    if args.partition:
        records = [r for r in records if args.partition in r["partition"].split(",")]
    if args.free:
        records = [r for r in records if r["gpus_usable"] > 0]

    def show(rows):
        _table(rows, [
            ("NODE", "node"),
            ("TYPE", "gpu_type"),
            ("STATUS", lambda r: r["status"] + ("+planned" if r["planned"] else "")),
            ("GPU_FREE", lambda r: f"{r['gpus_usable']}/{r['gpus_total']}"),
            ("FULL_SHARE", "gpus_full_share"),
            ("CPU_FREE", lambda r: f"{r['cpus_free']}/{r['cpus_total']}"),
            ("MEM_FREE", lambda r: f"{r['mem_free_gb']}G/{r['mem_total_gb']}G"),
            ("PARTITION", "partition"),
            ("USERS", lambda r: ",".join(f"{u}:{n}" for u, n in r["users"].items())),
        ])
        if any(r["planned"] for r in rows):
            print("\n+planned: the scheduler set this node's idle resources aside for a "
                  "pending higher-priority job;\nonly jobs that finish before it starts "
                  "(backfill) can use them.")
    _emit(args, records, show)


def _template_kwargs(args):
    return dict(gpus=args.gpus, nodes=args.nodes, cpus=args.cpus, mem_gb=args.mem,
                time=args.time, account=args.account, partition=args.partition,
                node=args.node, job_name=args.job_name, command=args.command)


def cmd_template(args):
    try:
        script = rq.slurm_template(args.gpu_type, **_template_kwargs(args))
    except ValueError as e:
        sys.exit(f"rres: {e}")
    if args.output:
        with open(args.output, "w") as f:
            f.write(script + "\n")
    if args.json:
        _emit(args, {"gpu_type": args.gpu_type, "path": args.output, "script": script}, None)
    elif args.output:
        print(f"wrote {args.output}; submit with: sbatch {args.output}")
    else:
        print(script)


def cmd_eta(args):
    results = []
    if args.script:
        with open(args.script) as f:
            est = rq.estimate_start(f.read())
        results.append({"gpu_type": None, "script": args.script, **est})
    else:
        alloc_data, _, free = rq.gpu_nodes_with_status()
        types = [args.gpu_type] if args.gpu_type else sorted(
            {s[0]["type"] for s in alloc_data.values() if s})
        for gtype in types:
            mem = max((rq.gpu_memory_gb(gtype, s[0]["features"]) or 0
                       for s in alloc_data.values() if s and s[0]["type"] == gtype),
                      default=0)
            if args.min_mem and mem < args.min_mem:
                continue
            try:
                script = rq.slurm_template(gtype, alloc_data, free, **_template_kwargs(args))
            except ValueError as e:
                results.append({"gpu_type": gtype, "error": str(e)})
                continue
            if "<your_account>" in script:
                sys.exit("rres: you have several accounts; pass --account")
            results.append({"gpu_type": gtype, **rq.estimate_start(script)})
    for r in results:
        r["wait_minutes"] = _wait_minutes(r.get("start"))
    results.sort(key=lambda r: (r["wait_minutes"] is None, r["wait_minutes"] or 0))

    def show(rows):
        ok = [r for r in rows if "error" not in r]
        _table(ok, [
            ("TYPE", lambda r: r["gpu_type"] or os.path.basename(r["script"])),
            ("WAIT", lambda r: _fmt_wait(r["wait_minutes"])),
            ("START", "start"),
            ("PARTITION", "partition"),
            ("NODES", "nodes"),
        ])
        for r in rows:
            if "error" in r:
                print(f"{r['gpu_type'] or r.get('script')}: {r['error']}")
        print("\nEstimates from `sbatch --test-only` (nothing was submitted). They assume "
              "running jobs use\ntheir whole time limit, so jobs usually start earlier; "
              "a shorter --time helps backfill.")
    _emit(args, results, show)


def cmd_jobs(args):
    jobs = rq.user_jobs(args.user)
    if args.state:
        jobs = [j for j in jobs if j["state"] == args.state.upper()]

    def show(rows):
        _table(rows, [
            ("JOBID", "jobid"),
            ("NAME", lambda r: r["name"][:30]),
            ("PARTITION", "partition"),
            ("ST", "state"),
            ("GPUS", lambda r: f"{r['gpus']}" + (f" {r['gpu_type']}" if r["gpu_type"] else "")),
            ("USED/LIMIT", lambda r: f"{r['time_used']}/{r['time_limit']}"),
            ("START", lambda r: r["start_time"] or "-"),
            ("NODES/REASON", lambda r: r["nodes"] if r["state"] == "R" else (r["reason"] or "-")),
        ])
    _emit(args, jobs, show)


def _dash(value):
    return "-" if value is None else value


def cmd_accounts(args):
    def show(rows):
        if not rows:
            print("no Slurm accounts found (sacctmgr unavailable or no associations)")
            return
        _table(rows, [
            ("ACCOUNT", "account"), ("BALANCE", lambda r: _dash(r["balance"])),
            ("RESERVED", lambda r: _dash(r["reserved"])),
            ("AVAILABLE", lambda r: _dash(r["available"])),
        ])
        if all(r["balance"] is None for r in rows):
            print("\nNo balances: this cluster has no `allocations` command.")
    _emit(args, rq.account_balances(), show)


def cmd_quota(args):
    quotas = rq.disk_quota()

    def show(rows):
        if rows is None:
            print("rres quota needs `hdquota`, which this cluster does not have;\n"
                  "try the site's own quota command (e.g. `quota -s`, `lfs quota`, `df -h ~`).")
            return
        _table(rows, [
            ("STORAGE", "storage"), ("LOCATION", "location"), ("SIZE", "size"),
            ("USED", "used"), ("AVAILABLE", "available"),
            ("USE%", lambda r: f"{r['use_percent']}%"),
        ])
    _emit(args, quotas, show)


# ------------------------------------------------------------------------ parser

def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print machine-readable JSON")

    job = argparse.ArgumentParser(add_help=False)
    g = job.add_argument_group("job shape (defaults follow the chosen node type)")
    g.add_argument("--gpus", type=int, default=1, help="GPUs per node (default: 1)")
    g.add_argument("--nodes", type=int, default=1, help="number of nodes (default: 1)")
    g.add_argument("--cpus", type=int, help="CPUs per task (default: the node's share per GPU x GPUs)")
    g.add_argument("--mem", type=int, metavar="GB", help="memory per node in GB (default: not requested, Slurm's default applies)")
    g.add_argument("--time", help=f"wall time, D-HH:MM:SS (default: {rq.DEFAULT_TIME}, capped at the partition limit)")
    g.add_argument("--account", help="Slurm account (default: yours, if you have only one)")
    g.add_argument("--partition", help="partition (default: the one holding the node type with the most free GPUs)")
    g.add_argument("--node", help="pin the job to this node (--nodelist)")
    g.add_argument("--job-name", default="my_job")
    g.add_argument("--command", default="python train.py", help="command to run in the script")

    parser = argparse.ArgumentParser(
        prog="rres", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")
    sub.required = True  # not a keyword argument before python 3.7

    p = sub.add_parser("gpus", parents=[common], help="free GPUs and queue length per GPU type")
    p.add_argument("--type", help="only this GPU type")
    p.add_argument("--min-mem", type=int, metavar="GB", help="only types with at least this much memory per GPU")
    p.set_defaults(func=cmd_gpus)

    p = sub.add_parser("nodes", parents=[common], help="per-node GPUs, CPUs, memory, status and users")
    p.add_argument("--type", help="only this GPU type")
    p.add_argument("--partition", help="only nodes in this partition")
    p.add_argument("--free", action="store_true", help="only nodes with usable free GPUs")
    p.set_defaults(func=cmd_nodes)

    p = sub.add_parser("template", parents=[common, job], help="generate an SBATCH script for a GPU type")
    p.add_argument("gpu_type", help="GPU type as listed by `rres gpus`, e.g. a100_80gb")
    p.add_argument("-o", "--output", help="write the script to this file")
    p.set_defaults(func=cmd_template)

    p = sub.add_parser("eta", parents=[common, job],
                       help="estimated start time per GPU type (sbatch --test-only, submits nothing)")
    p.add_argument("gpu_type", nargs="?", help="only this GPU type (default: every type)")
    p.add_argument("--script", help="estimate an existing job script instead")
    p.add_argument("--min-mem", type=int, metavar="GB", help="only types with at least this much memory per GPU")
    p.set_defaults(func=cmd_eta)

    p = sub.add_parser("jobs", parents=[common], help="your jobs (or another user's), with pending reasons and start estimates")
    p.add_argument("-u", "--user", help="user name (default: you)")
    p.add_argument("--state", help="only jobs in this state, e.g. R or PD")
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("accounts", parents=[common], help="your Slurm accounts (with SU balances where `allocations` exists)")
    p.set_defaults(func=cmd_accounts)

    p = sub.add_parser("quota", parents=[common], help="your storage quotas (needs `hdquota`)")
    p.set_defaults(func=cmd_quota)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    missing = [c for c in ("sinfo", "squeue", "scontrol") if not shutil.which(c)]
    if missing and args.cmd not in ("accounts", "quota"):
        sys.exit(f"rres: {', '.join(missing)} not found: run rres on a Slurm login or compute node")
    try:
        args.func(args)
    except subprocess.CalledProcessError as e:
        sys.exit(f"rres: `{e.cmd}` failed with exit status {e.returncode}")


if __name__ == "__main__":
    main()
