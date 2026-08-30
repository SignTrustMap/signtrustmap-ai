import os
import glob
import time
from datetime import datetime
from typing import Dict, List, Optional
import psutil
import torch
from loguru import logger
from src.api.schemas.system import (
    CPUMetrics,
    MemoryMetrics,
    GPUMetrics,
    ThermalMetrics,
    FanMetrics,
    PowerMetrics,
    DiskDeviceMetrics,
    NetworkMetrics,
    SystemMetricsResponse,
)


class SystemMetricsCollector:
    def __init__(self):
        self._boot_time = psutil.boot_time()
        self._last_time = time.time()
        self._last_net = psutil.net_io_counters()
        self._last_disk_io = psutil.disk_io_counters(perdisk=True)
        psutil.cpu_percent(interval=None)

    def _get_cpu(self) -> CPUMetrics:
        total_percent = psutil.cpu_percent(interval=None)
        per_core = psutil.cpu_percent(percpu=True)
        freq_info = psutil.cpu_freq()
        freq = round(freq_info.current, 1) if freq_info else None
        return CPUMetrics(
            total_percent=round(total_percent, 1),
            per_core_percent=[round(c, 1) for c in per_core],
            frequency_mhz=freq,
            core_count=psutil.cpu_count(logical=True) or len(per_core),
        )

    def _get_memory(self) -> MemoryMetrics:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return MemoryMetrics(
            total_mb=round(vm.total / (1024 * 1024), 1),
            used_mb=round(vm.used / (1024 * 1024), 1),
            free_mb=round(vm.free / (1024 * 1024), 1),
            available_mb=round(vm.available / (1024 * 1024), 1),
            percent=round(vm.percent, 1),
            swap_total_mb=round(swap.total / (1024 * 1024), 1),
            swap_used_mb=round(swap.used / (1024 * 1024), 1),
            swap_free_mb=round(swap.free / (1024 * 1024), 1),
            swap_percent=round(swap.percent, 1),
        )

    def _get_gpu(self) -> GPUMetrics:
        available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if available else None
        gpu_load = 0.0

        for path in ["/sys/devices/platform/17000000.ga10b/load", "/sys/devices/gpu.0/load"]:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        gpu_load = float(f.read().strip()) / 10.0
                    break
                except Exception:
                    pass

        alloc_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 1) if available else 0.0
        resv_mb = round(torch.cuda.memory_reserved() / (1024 * 1024), 1) if available else 0.0

        return GPUMetrics(
            available=available,
            device_name=device_name,
            load_percent=round(gpu_load, 1),
            memory_allocated_mb=alloc_mb,
            memory_reserved_mb=resv_mb,
        )

    def _get_thermal(self) -> ThermalMetrics:
        zones = {}
        for tz in glob.glob("/sys/class/thermal/thermal_zone*"):
            type_file = os.path.join(tz, "type")
            temp_file = os.path.join(tz, "temp")
            if os.path.exists(type_file) and os.path.exists(temp_file):
                try:
                    with open(type_file, "r", encoding="utf-8") as f:
                        zname = f.read().strip()
                    with open(temp_file, "r", encoding="utf-8") as f:
                        temp_c = float(f.read().strip()) / 1000.0
                    zones[zname] = round(temp_c, 1)
                except Exception:
                    pass

        cpu_temp = zones.get("cpu-thermal")
        gpu_temp = zones.get("gpu-thermal")
        soc_temp = zones.get("soc0-thermal", zones.get("soc-thermal"))
        tj_temp = zones.get("tj-thermal")

        return ThermalMetrics(
            cpu_temp_c=cpu_temp,
            gpu_temp_c=gpu_temp,
            soc_temp_c=soc_temp,
            tj_temp_c=tj_temp,
            all_zones=zones,
        )

    def _get_fan(self) -> FanMetrics:
        pwm_val = 0
        rpm_val = None

        for path in ["/sys/class/hwmon/hwmon0/pwm1", "/sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1"]:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        pwm_val = int(f.read().strip())
                    break
                except Exception:
                    pass

        for path in ["/sys/class/hwmon/hwmon3/rpm", "/sys/class/hwmon/hwmon3/fan1_input"]:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        rpm_val = int(f.read().strip())
                    break
                except Exception:
                    pass

        speed_pct = round((pwm_val / 255.0) * 100, 1)
        return FanMetrics(
            pwm=pwm_val,
            speed_percent=speed_pct,
            rpm=rpm_val,
        )

    def _get_power(self) -> PowerMetrics:
        voltage_v = None
        current_ma = None
        power_w = None

        ina_dir = "/sys/class/hwmon/hwmon1"
        in1_path = os.path.join(ina_dir, "in1_input")
        curr1_path = os.path.join(ina_dir, "curr1_input")

        if os.path.exists(in1_path):
            try:
                with open(in1_path, "r", encoding="utf-8") as f:
                    voltage_v = round(float(f.read().strip()) / 1000.0, 2)
            except Exception:
                pass

        if os.path.exists(curr1_path):
            try:
                with open(curr1_path, "r", encoding="utf-8") as f:
                    current_ma = round(float(f.read().strip()), 1)
            except Exception:
                pass

        if voltage_v is not None and current_ma is not None:
            power_w = round((voltage_v * current_ma) / 1000.0, 2)

        return PowerMetrics(
            voltage_v=voltage_v,
            current_ma=current_ma,
            power_w=power_w,
        )

    def _get_disks(self, delta_time: float) -> List[DiskDeviceMetrics]:
        current_disk_io = psutil.disk_io_counters(perdisk=True) or {}
        partitions = psutil.disk_partitions(all=False)
        disks = []
        seen_mounts = set()

        for part in partitions:
            if part.mountpoint in seen_mounts:
                continue
            if part.fstype in ["squashfs", "iso9660"]:
                continue
            seen_mounts.add(part.mountpoint)

            dev_name = os.path.basename(part.device)
            base_dev = dev_name.rstrip("0123456789p")
            if not base_dev:
                base_dev = dev_name

            read_speed = 0.0
            write_speed = 0.0

            matched_io_curr = current_disk_io.get(dev_name) or current_disk_io.get(base_dev)
            matched_io_last = self._last_disk_io.get(dev_name) or self._last_disk_io.get(base_dev)

            if matched_io_curr and matched_io_last and delta_time > 0:
                d_read = max(0, matched_io_curr.read_bytes - matched_io_last.read_bytes)
                d_write = max(0, matched_io_curr.write_bytes - matched_io_last.write_bytes)
                read_speed = round(d_read / delta_time / 1024.0, 1)
                write_speed = round(d_write / delta_time / 1024.0, 1)

            try:
                usage = psutil.disk_usage(part.mountpoint)
                disks.append(DiskDeviceMetrics(
                    device=part.device,
                    mountpoint=part.mountpoint,
                    fstype=part.fstype,
                    total_gb=round(usage.total / (1024 ** 3), 1),
                    used_gb=round(usage.used / (1024 ** 3), 1),
                    free_gb=round(usage.free / (1024 ** 3), 1),
                    percent=round(usage.percent, 1),
                    read_speed_kbps=read_speed,
                    write_speed_kbps=write_speed,
                ))
            except Exception:
                pass

        self._last_disk_io = current_disk_io
        return disks

    def _get_network(self, delta_time: float) -> NetworkMetrics:
        current_net = psutil.net_io_counters()
        down_speed = 0.0
        up_speed = 0.0

        if delta_time > 0 and self._last_net:
            d_recv = max(0, current_net.bytes_recv - self._last_net.bytes_recv)
            d_sent = max(0, current_net.bytes_sent - self._last_net.bytes_sent)
            down_speed = round(d_recv / delta_time / 1024.0, 1)
            up_speed = round(d_sent / delta_time / 1024.0, 1)

        self._last_net = current_net
        return NetworkMetrics(
            download_speed_kbps=down_speed,
            upload_speed_kbps=up_speed,
            total_recv_mb=round(current_net.bytes_recv / (1024 * 1024), 1),
            total_sent_mb=round(current_net.bytes_sent / (1024 * 1024), 1),
        )

    def collect_metrics(self) -> SystemMetricsResponse:
        now = time.time()
        delta_time = max(0.001, now - self._last_time)
        self._last_time = now

        uptime = round(now - self._boot_time, 1)
        iso_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cpu = self._get_cpu()
        memory = self._get_memory()
        gpu = self._get_gpu()
        thermal = self._get_thermal()
        fan = self._get_fan()
        power = self._get_power()
        disks = self._get_disks(delta_time)
        network = self._get_network(delta_time)

        return SystemMetricsResponse(
            timestamp=iso_time,
            uptime_seconds=uptime,
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            thermal=thermal,
            fan=fan,
            power=power,
            disks=disks,
            network=network,
        )


metrics_collector = SystemMetricsCollector()
