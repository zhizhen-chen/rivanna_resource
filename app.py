import os
from datetime import datetime
import argparse
import copy
import pytz
from collections import defaultdict
from subprocess import STDOUT, check_output
from flask import Flask, Response, render_template_string
from slurm_gpustat import (
    resource_by_type,
    parse_all_gpus,
    gpu_usage,
    gpu_alloc_by_node,
    node_states,
    INACCESSIBLE,
    DRAINING,
    parse_cmd,
    parse_squeue,
    avail_stats_for_node,
    parse_node_names,
    get_gpu_partitions,
)
# import gradio as gr  # Not used in this file


def get_resource_bar(avail, total, text="", long=False):
    """Create a long/short progress bar with text overlaid. Formatting handled in css."""

    if long:
        long_str = " class=long"
    else:
        long_str = ""
    if total == 0:
        total = 1  # avoid ZeroDivisionError Error
    bar = (
        f'<div class="progress" data-text="{text}">'
        f'<progress{long_str} max="100" value="{avail/total*100}"></progress></div>'
    )
    return bar


def str_to_int(text):
    """Convert string with unit to int in GB. e.g. "30 GB" --> 30, "1.4 T" --> 1400."""
    # Remove any whitespace
    text = text.strip()
    
    # Split into number and unit
    parts = text.split()
    if len(parts) != 2:
        return 0
    
    number = float(parts[0])
    unit = parts[1].upper()
    
    # Convert to GB
    if unit == 'T':
        return int(number * 1024)
    elif unit == 'G':
        return int(number)
    elif unit == 'M':
        return int(number / 1024)
    else:
        return 0


def parse_leaderboard(sum_by_gmem=[48]):
    """Request sinfo, parse the leaderboard in string."""

    resources = parse_all_gpus()
    usage = gpu_usage(
        resources=resources,
    )  # partition='gpu'
    aggregates = {}
    for user, subdict in usage.items():
        aggregates[user] = {}
        aggregates[user]["n_gpu"] = {
            key: sum([x["n_gpu"] for x in val.values()]) for key, val in subdict.items()
        }
        aggregates[user]["bash_gpu"] = {
            key: sum([x["bash_gpu"] for x in val.values()])
            for key, val in subdict.items()
        }
    out = ""
    for user, subdict in sorted(
        aggregates.items(), key=lambda x: sum(x[1]["n_gpu"].values()), reverse=True
    ):
        total = f"total={str(sum(subdict['n_gpu'].values())):2s}"
        user_summary = [
            f"{key}={val}"
            for key, val in sorted(subdict["n_gpu"].items(), key=lambda x: x[0])
        ]
        summary_str = "".join([f"{i:12s}" for i in user_summary])
        total += f"|bash={str(sum(subdict['bash_gpu'].values())):2s}"
        out += f"{user:12s}[{total}]    {summary_str}\n"
    return out


def parse_leaderboard_by_partition(sum_by_gmem=[48]):
    """Request sinfo, parse the leaderboard in string."""
    resources = parse_all_gpus()
    gpu_partitions = get_gpu_partitions()

    out = "=" * 64 + "\n"
    for i, part in enumerate(gpu_partitions):
        usage = gpu_usage(resources=resources, partition=part)  # partition='gpu'
        aggregates = {}
        for user, subdict in usage.items():
            aggregates[user] = {}
            aggregates[user]["n_gpu"] = {
                key: sum([x["n_gpu"] for x in val.values()])
                for key, val in subdict.items()
            }
            aggregates[user]["bash_gpu"] = {
                key: sum([x["bash_gpu"] for x in val.values()])
                for key, val in subdict.items()
            }
        if i != 0:
            out += "-" * 64 + "\n"
        out += f"PARTITION: {part}\n"
        for user, subdict in sorted(
            aggregates.items(), key=lambda x: sum(x[1]["n_gpu"].values()), reverse=True
        ):
            total = f"total={str(sum(subdict['n_gpu'].values())):2s}"
            user_summary = [
                f"{key}={val}"
                for key, val in sorted(subdict["n_gpu"].items(), key=lambda x: x[0])
            ]
            summary_str = "".join([f"{i:12s}" for i in user_summary])
            total += f"|bash={str(sum(subdict['bash_gpu'].values())):2s}"
            out += f"{user:12s}[{total}]    {summary_str}\n"
    out += "=" * 64 + "\n"
    return out


def cpu_usage(resources, partition="compute"):
    """Build a data structure of the CPU resource usage, organised by user.

    Args:
        resources (dict :: None): a summary of cluster resources, organised by node name.

    Returns:
        (dict): a summary of resources organised by user (and also by node name).
    """
    args = "-O NumNodes:100,nodelist:100,username:100,jobid:100 --noheader"
    if partition:
        args += f" --partition={partition}"
    rows = parse_squeue(args)
    usage = defaultdict(dict)
    for row in rows:
        tokens = row.split()
        # ignore pending jobs
        if len(tokens) < 4:
            continue
        cpu_count_str, node_str, user, jobid = tokens
        num_cpus = int(cpu_count_str.strip())
        node_names = parse_node_names(node_str)
        for node_name in node_names:
            # If a node still has jobs running but is draining, it will not be present
            # in the "available" resources, so we ignore it
            if node_name not in resources:
                continue
            cpu_type = resources[node_name]["type"]

            if cpu_type in usage[user]:
                usage[user][cpu_type][node_name]["n_cpu"] += num_cpus
            else:
                usage[user][cpu_type] = defaultdict(
                    lambda: {
                        "n_cpu": 0,
                    }
                )
                usage[user][cpu_type][node_name]["n_cpu"] += num_cpus
    return usage


def parse_cpu_usage_to_table(partition="compute", show_bar=True):
    """Request sinfo for cnode, parse the output to a html table."""

    node_str = parse_cmd(f"sinfo -o '%1000N' --noheader --partition={partition}")
    assert isinstance(node_str, list) and len(node_str) == 1
    node_names = parse_node_names(node_str[0].strip())
    resources = {k: {"type": k[0:6] + "xx", "count": 1} for k in node_names}
    states = node_states(partition=partition)
    res = {
        key: val
        for key, val in resources.items()
        if states.get(key, "down") not in INACCESSIBLE
    }
    res_total = copy.deepcopy(res)
    usage = cpu_usage(resources=res, partition=partition)

    for subdict in usage.values():
        for cpu_type, node_dicts in subdict.items():
            for node_name, user_cpu_count in node_dicts.items():
                count = res[node_name]["count"]
                count = max(count - user_cpu_count["n_cpu"], 0)
                res[node_name]["count"] = count

    res_total_by_type = defaultdict(list)
    for node, spec in res_total.items():
        res_total_by_type[spec["type"]].append({"node": node, "count": spec["count"]})

    res_usage_by_type = defaultdict(list)
    for node, spec in res.items():
        res_usage_by_type[spec["type"]].append({"node": node, "count": spec["count"]})

    table_html = []
    total_cpu_count = 0
    avail_cpu_count = 0

    # sort cpus from new to old
    type_list = sorted(list(res_total_by_type.keys()), reverse=True)

    # writing the html table
    for cpu_type in type_list:
        node_dicts = res_total_by_type[cpu_type]
        node_names = sorted([i["node"] for i in node_dicts])

        node_summaries = []
        num_col = []

        for node in node_names:
            node_name = f"<td>{node}</td>"

            users = [user for user in usage if node in usage[user].get(cpu_type, [])]
            if len(users):
                users = f"<td>user: {','.join(users)}</td>"
            else:
                users = f"<td>&nbsp</td>"

            detail_dict = avail_stats_for_node(node)
            detail_dict = {k: v for k, v in detail_dict.items() if k in ["cpu", "mem"]}

            if show_bar:
                c_stat = detail_dict["cpu"].split("/")
                c_stat = [int(i.strip()) for i in c_stat]
                cpu_bar = get_resource_bar(*c_stat, text=detail_dict["cpu"])

                m_stat = detail_dict["mem"].split("/")
                m_stat = [str_to_int(i.strip()) for i in m_stat]
                mem_bar = get_resource_bar(*m_stat, text=detail_dict["mem"], long=True)

                total_cpu_count += c_stat[-1]
                avail_cpu_count += c_stat[-1] - c_stat[0]
            else:
                cpu_bar = detail_dict["cpu"]
                mem_bar = detail_dict["mem"]
            cpu_stat = f"<td>cpu: {cpu_bar}</td>"
            mem_stat = f"<td>mem: {mem_bar}</td>"

            node_summary = (
                f"<tr><td>&nbsp</td>{node_name}{cpu_stat}{mem_stat}{users}</tr>"
            )
            node_summaries.append(node_summary)
            num_col.append(5)

        type_summary = (
            f'<tr><td colspan="{max(num_col)}"><b>' f"{cpu_type}: </b></td></tr>"
        )
        table_html.append(type_summary)
        table_html.extend(node_summaries)

    if show_bar:
        total_bar = get_resource_bar(
            avail_cpu_count,
            total_cpu_count,
            text=f"{avail_cpu_count} / {total_cpu_count}",
        )
        total_summary = (
            f'<tr><td colspan="{max(num_col)}"><h3>'
            f"Summary: {total_bar} cpus available</h3></td></tr>"
        )
    else:
        total_bar = f"{avail_cpu_count}/{total_cpu_count}"
        total_summary = ""

    table_html = f"<table>{total_summary}{''.join(table_html)}</table>"

    return table_html


def build_slurm_template(gpu_type: str, alloc_data: dict) -> str:
    """Generate a SBATCH script header for the given GPU type.

    Picks the representative node (most common config) and fills in
    partition, constraint/feature, GPU name, and per-node CPU/mem totals.
    Values the user should customise (num GPUs, CPUs, mem, time) are left
    with placeholder comments so the intent is obvious.
    """
    nodes = [
        specs[0] for specs in alloc_data.values()
        if specs and specs[0]["type"] == gpu_type
    ]
    if not nodes:
        return f"# No nodes found for GPU type: {gpu_type}"

    # Use first node as representative (they share the same hardware config)
    rep = nodes[0]
    partition = rep.get("partition", "gpu")
    features = rep.get("features", gpu_type)
    active_features = rep.get("active_features", features)
    cpu_total = rep.get("cpu_total", 0)
    mem_total_mb = rep.get("mem_total_mb", 0)
    gpu_total = rep.get("total", 1)

    # ActiveFeatures is what SLURM enforces for --constraint; pick the most
    # specific tag (longest) that matches the gpu_type by normalised comparison
    # (ignoring underscores), so e.g. "rtxpro6000" is chosen for gpu_type
    # "rtx_pro_6000", and "a100_80gb" is chosen over "a100" for 80 GB nodes.
    active_feature_list = [f.strip() for f in active_features.split(",") if f.strip()]
    gpu_type_norm = gpu_type.replace("_", "")
    matching_features = [f for f in active_feature_list
                         if f.replace("_", "") == gpu_type_norm
                         or gpu_type_norm.startswith(f.replace("_", ""))]
    constraint_tag = max(matching_features, key=len) if matching_features else gpu_type

    # Derive the base gres name from AvailableFeatures: if it contains the bare
    # GPU name without a variant suffix, use that; otherwise use gpu_type.
    # e.g. features="a100,a100_80gb,gpupod" -> gres_name="a100"
    feature_list = [f.strip() for f in features.split(",") if f.strip()]
    bare_candidates = [f for f in feature_list
                       if gpu_type.startswith(f + "_") or f == gpu_type.split("_")[0]]
    gres_name = bare_candidates[0] if bare_candidates else gpu_type

    mem_per_gpu_gb = (mem_total_mb // 1024) // gpu_total if gpu_total else 0
    cpu_per_gpu = cpu_total // gpu_total if gpu_total else 0

    lines = [
        "#!/bin/bash",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --gres=gpu:{gres_name}:1          # number of GPUs (max {gpu_total} per node)",
        f"#SBATCH --constraint={constraint_tag}",
        f"#SBATCH --cpus-per-task={cpu_per_gpu}    # CPUs per GPU (max {cpu_total} per node)",
        f"#SBATCH --mem={mem_per_gpu_gb}G           # memory per job (max {mem_total_mb // 1024}G per node)",
        "#SBATCH --time=1-00:00:00                  # wall time (D-HH:MM:SS)",
        "#SBATCH --account=<your_account>           # your Slurm account/allocation",
        "#SBATCH --job-name=my_job",
        "#SBATCH --output=%x-%j.out",
        "",
        "# ---- load modules / activate environment ----",
        "module purge",
        "# module load cuda/12.x",
        "",
        "# ---- your commands ----",
        "python train.py",
    ]
    return "\n".join(lines)


def get_slurm_template_json(gpu_type: str) -> dict:
    """Return the SBATCH template and node-count info for a GPU type as a dict."""
    alloc_data = gpu_alloc_by_node()
    nodes = [s[0] for s in alloc_data.values() if s and s[0]["type"] == gpu_type]
    template = build_slurm_template(gpu_type, alloc_data)
    avail = sum(max(s[0]["total"] - s[0]["alloc"], 0) for s in alloc_data.values()
                if s and s[0]["type"] == gpu_type)
    total = sum(s[0]["total"] for s in alloc_data.values()
                if s and s[0]["type"] == gpu_type)
    return {"gpu_type": gpu_type, "avail": avail, "total": total, "template": template}


def parse_usage_to_table(show_bar=True):
    """Request sinfo/scontrol, parse the output to a html table."""

    # scontrol AllocTRES is the authoritative source for GPU allocation;
    # it covers all partitions without a hardcoded list.
    alloc_data = gpu_alloc_by_node()
    states = node_states()

    # Filter out inaccessible nodes.  Draining nodes are kept: they still run the
    # jobs already on them, so dropping them makes running work vanish from the
    # page.  They are shown greyed out with zero availability instead.
    draining = {
        node for node in alloc_data if states.get(node, "down") in DRAINING
    }
    alloc_data = {
        node: specs
        for node, specs in alloc_data.items()
        if node in draining or states.get(node, "down") not in INACCESSIBLE
    }

    # Build res_total (total GPUs) and res (available GPUs) in the same shape
    # as the old parse_all_gpus() output so the rest of the function is unchanged.
    res_total = {
        node: [{"type": s["type"], "count": s["total"]} for s in specs]
        for node, specs in alloc_data.items()
    }
    res = {
        node: [{"type": s["type"],
                "count": 0 if node in draining else max(s["total"] - s["alloc"], 0)}
               for s in specs]
        for node, specs in alloc_data.items()
    }

    # gpu_usage() is still used to get user-per-node info (squeue, no partition filter)
    usage = gpu_usage(resources=res_total)

    res_total_by_type = resource_by_type(res_total)
    res_usage_by_type = resource_by_type(res)

    type_blocks = []
    total_gpu_count = 0
    avail_gpu_count = 0

    for gpu_type in sorted(res_total_by_type.keys()):
        node_dicts = res_total_by_type[gpu_type]
        node_names = sorted(i["node"] for i in node_dicts)
        gpu_count_total = {i["node"]: i["count"] for i in node_dicts}
        gpu_count_avail = {i["node"]: i["count"] for i in res_usage_by_type[gpu_type]}

        node_rows = []
        for node in node_names:
            if node in draining:
                node_cell = (
                    f'{node}<span class="draining-tag" title="draining: the jobs '
                    f'already running continue, but no new job is accepted">'
                    f'draining</span>'
                )
            else:
                node_cell = node

            detail = avail_stats_for_node(node)
            if show_bar:
                gpu_bar = get_resource_bar(
                    gpu_count_avail[node],
                    gpu_count_total[node],
                    text=f"{gpu_count_avail[node]} / {gpu_count_total[node]}",
                )
                cpu_bar = get_resource_bar(
                    *[int(i.strip()) for i in detail["cpu"].split("/")],
                    text=detail["cpu"],
                )
                mem_bar = get_resource_bar(
                    *[str_to_int(i.strip()) for i in detail["mem"].split("/")],
                    text=detail["mem"],
                    long=True,
                )
            else:
                gpu_bar = f"{gpu_count_avail[node]}/{gpu_count_total[node]}"
                cpu_bar, mem_bar = detail["cpu"], detail["mem"]

            users = [user for user in usage if node in usage[user].get(gpu_type, [])]
            user_cell = ",".join(users)

            row_class = ' class="draining"' if node in draining else ""
            node_rows.append(
                f"<tr{row_class}>"
                f'<td class="c-node">{node_cell}</td>'
                f'<td><span class="c-label">gpu</span> {gpu_bar}</td>'
                f'<td><span class="c-label">cpu</span> {cpu_bar}</td>'
                f'<td><span class="c-label">mem</span> {mem_bar}</td>'
                f'<td class="c-users" title="{user_cell}">{user_cell or "&nbsp;"}</td>'
                "</tr>"
            )

        type_avail, type_total = sum(gpu_count_avail.values()), sum(gpu_count_total.values())
        if show_bar:
            type_bar = get_resource_bar(
                type_avail, type_total, text=f"{type_avail} / {type_total}"
            )
        else:
            type_bar = f"{type_avail}/{type_total}"
        gpu_label = (
            f'<a href="#" onclick="showSlurmTemplate(\'{gpu_type}\'); return false;"'
            f' title="Click to generate SBATCH template">{gpu_type}</a>'
        )
        type_blocks.append(
            f'<div class="gpu-type"><div class="gpu-type-head">{gpu_label}{type_bar}'
            f'<span class="c-label">gpus available</span></div>'
            f'<table><tbody>{"".join(node_rows)}</tbody></table></div>'
        )
        total_gpu_count += type_total
        avail_gpu_count += type_avail

    if show_bar:
        total_bar = get_resource_bar(
            avail_gpu_count,
            total_gpu_count,
            text=f"{avail_gpu_count} / {total_gpu_count}",
        )
    else:
        total_bar = f"{avail_gpu_count}/{total_gpu_count}"
    total_summary = (
        f'<div class="gpu-total">Summary{total_bar}'
        f'<span class="c-label">gpus available</span></div>'
    )
    return total_summary + "".join(type_blocks)


def parse_disk_io():
    """Measure disk reading speed, parse the output to a html table.

    Pre-requisite: create a byte file by running
    `dd if=/dev/zero of=/your/path/test.img bs=512MB count=1 oflag=dsync`."""

    # return '<p>Under maintenance. </p>'

    try:
        beegfs_ultra_read = check_output(
            "dd if=/scratch/shared/beegfs/shared-datasets/test/test.img of=/dev/null bs=512MB count=1 oflag=dsync",
            stderr=STDOUT,
            shell=True,
        ).decode("utf-8")
        beegfs_ultra_read = beegfs_ultra_read.split("\n")[-2].split(",")[-1].strip()
    except:
        beegfs_ultra_read = "N/A"

    try:
        beegfs_fast_read = check_output(
            "dd if=/scratch/shared/beegfs/htd/DATA/tmp/test.img of=/dev/null bs=512MB count=1 oflag=dsync",
            stderr=STDOUT,
            shell=True,
        ).decode("utf-8")
        beegfs_fast_read = beegfs_fast_read.split("\n")[-2].split(",")[-1].strip()
    except:
        beegfs_fast_read = "N/A"

    try:
        beegfs_normal_read = check_output(
            "dd if=/scratch/shared/beegfs/htd/tmp/test.img of=/dev/null bs=512MB count=1 oflag=dsync",
            stderr=STDOUT,
            shell=True,
        ).decode("utf-8")
        beegfs_normal_read = beegfs_normal_read.split("\n")[-2].split(",")[-1].strip()
    except:
        beegfs_normal_read = "N/A"

    try:
        work_normal_read = check_output(
            "dd if=/work/htd/Desktop_tmp/tmp/test.img of=/dev/null bs=512MB count=1 oflag=dsync",
            stderr=STDOUT,
            shell=True,
        ).decode("utf-8")
        work_normal_read = work_normal_read.split("\n")[-2].split(",")[-1].strip()
    except:
        work_normal_read = "N/A"

    summary = (
        "<tr> <td><b>{}</b></td> <td><b>{}</b></td> <td><b>{}</b></td> </tr>".format(
            "Disk", "Type", "Read Speed"
        )
    )
    summary += "<tr> <td>{}</td> <td>{}</td> <td>{}</td> </tr>".format(
        "/beegfs/shared-datasets <i>[ultra-fast-layer]</i>",
        "NVMe flash",
        beegfs_ultra_read,
    )
    summary += "<tr> <td>{}</td> <td>{}</td> <td>{}</td> </tr>".format(
        "/beegfs <i>[fast-layer]</i>", "SSD flash", beegfs_fast_read
    )
    summary += "<tr> <td>{}</td> <td>{}</td> <td>{}</td> </tr>".format(
        "/beegfs <i>[normal-layer]</i>", "HDD", beegfs_normal_read
    )
    summary += "<tr> <td>{}</td> <td>{}</td> <td>{}</td></tr>".format(
        "/work", "HDD", work_normal_read
    )
    table_html = f"<table>{summary}</table>"

    return table_html


def parse_disk_quota():
    """Run 'hdquota' command and parse the output to an HTML table."""
    try:
        output = check_output("hdquota", shell=True).decode("utf-8")
        lines = output.split('\n')
        
        # Create HTML table with headers
        headers = ["Storage Type", "Location", "Size", "Used", "Avail", "Use%"]
        table_html = "<table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
        
        # Add data rows
        for line in lines[2:]:  # Skip header and separator lines
            if not line.strip():
                continue
            
            # Split the line into parts, handling spaces correctly
            parts = line.split()
            
            # Skip lines that don't have enough parts
            if len(parts) < 9:  # Changed from 10 to 9 to handle single-word storage types
                continue
                
            try:
                # Handle storage type (could be one or two words)
                if parts[1] in ["Directory", "Project", "Standard"]:
                    storage_type = f"{parts[0]} {parts[1]}"
                    location = parts[2]
                    size = f"{parts[3]} {parts[4]}"
                    used = f"{parts[5]} {parts[6]}"
                    avail = f"{parts[7]} {parts[8]}"
                    use_percent = parts[9]
                else:
                    storage_type = parts[0]
                    location = parts[1]
                    size = f"{parts[2]} {parts[3]}"
                    used = f"{parts[4]} {parts[5]}"
                    avail = f"{parts[6]} {parts[7]}"
                    use_percent = parts[8]
                
                table_html += "<tr>"
                table_html += f"<td>{storage_type}</td>"
                table_html += f"<td>{location}</td>"
                table_html += f"<td>{size}</td>"
                table_html += f"<td>{used}</td>"
                table_html += f"<td>{avail}</td>"
                table_html += f"<td>{use_percent}</td>"
                table_html += "</tr>"
            except IndexError:
                continue
        
        table_html += "</table>"
        return table_html
    except Exception as e:
        error_msg = f"Error getting disk quota information: {str(e)}"
        return f"<p>{error_msg}</p>"


def get_my_accounts():
    """Return the list of accounts the current user belongs to via 'allocations'."""
    try:
        raw = parse_cmd("allocations", split=False)
    except Exception:
        return []
    accounts = []
    for line in raw.splitlines()[2:]:  # skip header and separator
        if not line.strip():
            break  # blank line marks end of account table
        accounts.append(line.split()[0])
    return accounts


def parse_allocation_for_account(account):
    """Parse 'allocations -a <account>' and return an HTML block for that account."""
    try:
        raw = check_output(f"allocations -a {account}", shell=True).decode("utf-8")
    except Exception:
        return f"<p>Error fetching allocation for {account}.</p>"

    if not raw.strip():
        return f"<p>No allocation information found for {account}.</p>"

    # Split into sections on blank lines; skip header/separator lines
    sections = []
    current = []
    for line in raw.splitlines():
        if "------" in line:
            continue
        if not line.strip():
            if current:
                sections.append(current)
                current = []
        else:
            current.append(line)
    if current:
        sections.append(current)

    # sections[0] = allocation rows (with header), sections[1] = member rows (with header)
    html = f"<h3>Allocations for {account}</h3>"

    if len(sections) >= 1:
        rows = sections[0]
        if len(rows) >= 2:  # first row is header
            headers = rows[0].split()
            html += "<table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
            for row in rows[1:]:
                # Use separator line from raw to detect column boundaries
                sep_line = next((l for l in raw.splitlines() if l.strip().startswith("---")), None)
                if sep_line:
                    cols = []
                    start = 0
                    for part in sep_line.split(" "):
                        if part:
                            cols.append((start, start + len(part)))
                        start += len(part) + 1
                    cells = [(row[s:e] if e <= len(row) else row[s:]).strip() for s, e in cols]
                else:
                    cells = row.split()
                html += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
            html += "</table><br>"

    if len(sections) >= 2:
        rows = sections[1]
        if len(rows) >= 2:
            headers = rows[0].split()
            html += "<table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
            for row in rows[1:]:
                if row.strip().startswith("PI:"):
                    continue
                # CommonName may be two words; EmailAddress and DefaultAccount follow
                parts = row.split()
                # Detect two-word common name: parts[2] is not an email
                if len(parts) >= 6 and "@" not in parts[2]:
                    cells = [parts[0], parts[1], f"{parts[2]} {parts[3]}", parts[4], parts[5]]
                elif len(parts) >= 5:
                    cells = parts[:5]
                else:
                    cells = parts
                html += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
            html += "</table>"

    return html


def parse_allocations_to_table():
    """Dynamically discover accounts and render allocation tables for each."""
    accounts = get_my_accounts()
    if not accounts:
        return "<p>No allocations found.</p>"
    return "".join(parse_allocation_for_account(acc) for acc in accounts)


def main():
    parser = argparse.ArgumentParser(description="launch web app")
    parser.add_argument(
        "--host", default="0.0.0.0", help="the host address for the website"
    )
    parser.add_argument(
        "--port", default=2070, type=int, help="the port for the website"
    )
    args = parser.parse_args()

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(open("index.html").read(), hostname=args.host)

    @app.route("/time_feed")
    def time_feed():
        def generate():
            yield f'updated at: {datetime.now(pytz.timezone("America/New_York")).strftime("%Y.%m.%d | %H:%M:%S")}'

        return Response(generate(), mimetype="text")

    @app.route("/resource")
    def resource():
        def generate():
            out = parse_usage_to_table()
            yield out

        return Response(generate(), mimetype="text")

    @app.route("/leaderboard")
    def leaderboard():
        def generate():
            out = parse_leaderboard()
            yield out

        return Response(generate(), mimetype="text")

    @app.route("/leaderboard_partition")
    def leaderboard_partition():
        def generate():
            out = parse_leaderboard_by_partition()
            yield out

        return Response(generate(), mimetype="text")

    @app.route("/disk_quota")
    def disk_quota():
        def generate():
            out = parse_disk_quota()
            yield out
        return Response(generate(), mimetype="text")

    @app.route("/allocations")
    def allocations():
        def generate():
            out = parse_allocations_to_table()
            yield out

        return Response(generate(), mimetype="text")

    @app.route("/slurm_template/<gpu_type>")
    def slurm_template(gpu_type):
        import json
        data = get_slurm_template_json(gpu_type)
        return Response(json.dumps(data), mimetype="application/json")

    # @app.route('/cpu_resource')
    # def cpu_resource():
    #     def generate():
    #         out = parse_cpu_usage_to_table()
    #         yield out
    #     return Response(generate(), mimetype='text')

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
