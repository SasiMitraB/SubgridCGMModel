#!/usr/bin/env python3
"""
System Resource & Task Monitor
- System overall: CPU, GPU Compute, RAM, and VRAM usage.
- Per-process tracking: Shows CPU, RAM, GPU Compute, and VRAM usage for each active
  Athena / AthenaK simulation or ML / Deep Learning training workload.
"""

import os
import re
import sys
import time
import argparse
import subprocess


def get_cpu_usage(interval=0.2):
    """Calculate overall CPU usage percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=interval)
    except ImportError:
        pass

    def read_cpu_times():
        with open("/proc/stat", "r") as f:
            line = f.readline()
        parts = [float(x) for x in line.split()[1:8]]
        idle = parts[3] + parts[4]
        total = sum(parts)
        return idle, total

    try:
        idle1, total1 = read_cpu_times()
        time.sleep(interval)
        idle2, total2 = read_cpu_times()
        idle_delta = idle2 - idle1
        total_delta = total2 - total1
        if total_delta <= 0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))
    except Exception:
        return None


def get_ram_usage():
    """Get overall RAM usage in GB and percentage."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "used_gb": mem.used / (1024 ** 3),
            "total_gb": mem.total / (1024 ** 3),
            "percent": mem.percent
        }
    except ImportError:
        pass

    try:
        meminfo = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    meminfo[key] = float(val)
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", 0)
        used_kb = total_kb - avail_kb
        percent = (used_kb / total_kb * 100) if total_kb > 0 else 0
        return {
            "used_gb": used_kb / (1024 ** 2),
            "total_gb": total_kb / (1024 ** 2),
            "percent": percent
        }
    except Exception:
        return None


def get_gpu_usage():
    """Query NVIDIA GPU usage and total VRAM."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            idx, name, util, mem_used, mem_total = [x.strip() for x in line.split(",")]
            used_mb = float(mem_used)
            total_mb = float(mem_total)
            mem_pct = (used_mb / total_mb * 100) if total_mb > 0 else 0
            gpus.append({
                "index": idx,
                "name": name,
                "gpu_util_pct": float(util),
                "vram_used_gb": used_mb / 1024,
                "vram_total_gb": total_mb / 1024,
                "vram_pct": mem_pct
            })
        return gpus
    except Exception:
        return []


def get_active_tasks():
    """Detect active Athena simulations and ML training runs with per-process CPU, RAM, GPU, and VRAM."""
    my_pid = str(os.getpid())

    # 1. Per-process GPU SM compute utilization via nvidia-smi pmon
    gpu_sm = {}
    try:
        pmon = subprocess.run(["nvidia-smi", "pmon", "-c", "1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in pmon.stdout.strip().split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5:
                p_id = parts[1]
                sm = parts[3]
                gpu_sm[p_id] = (sm + "%") if sm != "-" else "0%"
    except Exception:
        pass

    # 2. Per-process VRAM memory usage via nvidia-smi compute-apps
    gpu_vram = {}
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for line in res.stdout.strip().split("\n"):
            if line.strip():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 2:
                    mb = float(parts[1])
                    gpu_vram[parts[0]] = f"{mb/1024:.2f} GB" if mb >= 1024 else f"{int(mb)} MB"
    except Exception:
        pass

    # 3. System process list via ps
    try:
        res = subprocess.run(
            ["ps", "-eo", "pid,user,%cpu,%mem,rss,args"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        lines = res.stdout.strip().split("\n")
    except Exception:
        return []

    detected = []
    ignored_patterns = [
        "codeium", "language_server", "extensionHost", "node", "gitstatusd", "btop", "grep"
    ]

    for line in lines[1:]:
        parts = line.strip().split(None, 5)
        if len(parts) < 6:
            continue
        pid, user, cpu, mem_pct, rss, cmd = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]

        if pid == my_pid:
            continue
        if any(ign in cmd for ign in ignored_patterns):
            continue

        # Check Athena simulation
        is_athena = bool(
            re.search(r"(?:^|[\s/])(athena|athenak|athinput)\b", cmd, re.IGNORECASE) or
            ("prterun" in cmd and "athena" in cmd) or
            ("mpirun" in cmd and "athena" in cmd)
        )

        # Check ML Training
        is_ml = False
        if re.search(r"(?:python|torchrun|accelerate|deepspeed|train_)", cmd, re.IGNORECASE):
            if re.search(r"(train|fit|torch|tensorflow|jax|keras|lightning|wandb|model)", cmd, re.IGNORECASE):
                if "monitor_usage.py" not in cmd:
                    is_ml = True

        is_gpu = pid in gpu_vram or pid in gpu_sm

        if is_athena or is_ml or is_gpu:
            category_tags = []
            if is_athena:
                category_tags.append("Athena")
            if is_ml:
                category_tags.append("ML Training")
            if is_gpu and not category_tags:
                category_tags.append("GPU Compute")

            rss_mb = float(rss) / 1024
            ram_str = f"{rss_mb/1024:.2f} GB" if rss_mb >= 1024 else f"{rss_mb:.1f} MB"

            detected.append({
                "pid": pid,
                "user": user,
                "category": "/".join(category_tags),
                "cpu": f"{cpu}%",
                "ram": f"{ram_str} ({mem_pct}%)",
                "gpu": gpu_sm.get(pid, "0%"),
                "vram": gpu_vram.get(pid, "-"),
                "cmd": cmd
            })

    return detected


def format_bar(percent, length=20):
    """Generate a visual text progress bar."""
    filled = int(length * percent / 100)
    filled = max(0, min(length, filled))
    return f"[{'#' * filled}{'.' * (length - filled)}] {percent:5.1f}%"


def display_stats(cpu_interval=0.2):
    cpu = get_cpu_usage(interval=cpu_interval)
    ram = get_ram_usage()
    gpus = get_gpu_usage()
    tasks = get_active_tasks()

    width = 95
    lines = []
    lines.append("=" * width)
    lines.append(f"  SYSTEM RESOURCE MONITOR - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * width)

    # Global CPU
    if cpu is not None:
        lines.append(f"  CPU Total   : {format_bar(cpu)} ({os.cpu_count()} vCPUs)")
    else:
        lines.append("  CPU Total   : N/A")

    # Global RAM
    if ram is not None:
        ram_bar = format_bar(ram["percent"])
        lines.append(f"  RAM Total   : {ram_bar} ({ram['used_gb']:.2f} / {ram['total_gb']:.2f} GB)")
    else:
        lines.append("  RAM Total   : N/A")

    # Global GPU / VRAM
    if gpus:
        lines.append("-" * width)
        for gpu in gpus:
            lines.append(f"  GPU {gpu['index']} [{gpu['name']}]:")
            lines.append(f"    GPU Compute : {format_bar(gpu['gpu_util_pct'])}")
            vram_bar = format_bar(gpu['vram_pct'])
            lines.append(f"    VRAM Usage  : {vram_bar} ({gpu['vram_used_gb']:.2f} / {gpu['vram_total_gb']:.2f} GB)")
    else:
        lines.append("-" * width)
        lines.append("  GPU / VRAM  : No NVIDIA GPU detected or nvidia-smi unavailable")

    # Athena & ML Tasks Section
    lines.append("-" * width)
    lines.append("  ACTIVE PROCESS USAGE (Athena / ML Training / GPU Workloads):")
    if tasks:
        lines.append(f"  {'USER':<9} {'TYPE':<12} {'PID':<8} {'CPU%':<8} {'RAM (RSS)':<16} {'GPU%':<8} {'VRAM':<10} COMMAND")
        lines.append(f"  {'-'*7}   {'-'*10}   {'-'*6}   {'-'*6}   {'-'*14}   {'-'*6}   {'-'*8}   {'-'*20}")
        for t in tasks:
            cmd_preview = t['cmd']
            if len(cmd_preview) > 55:
                cmd_preview = cmd_preview[:52] + "..."
            lines.append(f"  {t['user']:<9} {t['category']:<12} {t['pid']:<8} {t['cpu']:<8} {t['ram']:<16} {t['gpu']:<8} {t['vram']:<10} {cmd_preview}")
    else:
        lines.append("  No active Athena simulations or ML training processes detected.")

    lines.append("=" * width)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Monitor CPU, GPU, RAM, VRAM, and Athena/ML workloads.")
    parser.add_argument("-w", "--watch", action="store_true", help="Live monitor continuously")
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="Refresh interval in seconds (default: 1.0)")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                sys.stdout.write("\033[H\033[J")
                print(display_stats(cpu_interval=min(0.2, args.interval)))
                time.sleep(max(0.0, args.interval - 0.2))
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print(display_stats(cpu_interval=0.2))


if __name__ == "__main__":
    main()
