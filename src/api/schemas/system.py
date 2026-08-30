from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CPUMetrics(BaseModel):
    total_percent: float = Field(..., description="Overall CPU usage percentage (0.0 - 100.0)", examples=[15.4])
    per_core_percent: List[float] = Field(..., description="CPU usage percentage per individual core", examples=[[12.1, 18.3, 10.0, 14.5, 22.0, 8.5]])
    frequency_mhz: Optional[float] = Field(None, description="Current CPU frequency in MHz", examples=[1420.0])
    core_count: int = Field(..., description="Total number of logical CPU cores", examples=[6])


class MemoryMetrics(BaseModel):
    total_mb: float = Field(..., description="Total physical RAM in MB", examples=[7620.0])
    used_mb: float = Field(..., description="Used RAM in MB", examples=[3850.0])
    free_mb: float = Field(..., description="Free unused RAM in MB", examples=[2310.0])
    available_mb: float = Field(..., description="RAM available for allocation in MB", examples=[3350.0])
    percent: float = Field(..., description="RAM usage percentage", examples=[50.5])
    swap_total_mb: float = Field(..., description="Total Swap in MB", examples=[3810.0])
    swap_used_mb: float = Field(..., description="Used Swap in MB", examples=[1100.0])
    swap_free_mb: float = Field(..., description="Free Swap in MB", examples=[2710.0])
    swap_percent: float = Field(..., description="Swap usage percentage", examples=[28.8])


class GPUMetrics(BaseModel):
    available: bool = Field(..., description="Whether CUDA GPU is available", examples=[True])
    device_name: Optional[str] = Field(None, description="Name of the GPU hardware", examples=["Orin"])
    load_percent: float = Field(..., description="GPU utilization percentage (0.0 - 100.0)", examples=[45.0])
    memory_allocated_mb: float = Field(..., description="CUDA memory currently allocated in MB", examples=[320.5])
    memory_reserved_mb: float = Field(..., description="CUDA memory reserved by PyTorch caching allocator in MB", examples=[512.0])


class ThermalMetrics(BaseModel):
    cpu_temp_c: Optional[float] = Field(None, description="CPU temperature in degrees Celsius", examples=[48.2])
    gpu_temp_c: Optional[float] = Field(None, description="GPU temperature in degrees Celsius", examples=[47.5])
    soc_temp_c: Optional[float] = Field(None, description="System-on-Chip board temperature in degrees Celsius", examples=[46.8])
    tj_temp_c: Optional[float] = Field(None, description="Tjunction temperature in degrees Celsius", examples=[47.9])
    all_zones: Dict[str, float] = Field(default_factory=dict, description="Dictionary of all thermal zone temperatures")


class FanMetrics(BaseModel):
    pwm: int = Field(..., description="Fan PWM raw value (0 - 255)", examples=[178])
    speed_percent: float = Field(..., description="Fan speed percentage (0.0 - 100.0%)", examples=[69.8])
    rpm: Optional[int] = Field(None, description="Actual fan rotation speed in RPM", examples=[4225])


class PowerMetrics(BaseModel):
    voltage_v: Optional[float] = Field(None, description="Input voltage in Volts (VDD_IN)", examples=[5.0])
    current_ma: Optional[float] = Field(None, description="Current draw in mA", examples=[950.0])
    power_w: Optional[float] = Field(None, description="Instantaneous power consumption in Watts", examples=[4.75])


class DiskDeviceMetrics(BaseModel):
    device: str = Field(..., description="Disk device name (e.g., sda3, nvme0n1p1)", examples=["sda3"])
    mountpoint: str = Field(..., description="Filesystem mount path", examples=["/media/taiduc_orico_ssd"])
    fstype: str = Field(..., description="Filesystem type", examples=["ext4"])
    total_gb: float = Field(..., description="Total disk space in GB", examples=[953.8])
    used_gb: float = Field(..., description="Used disk space in GB", examples=[420.5])
    free_gb: float = Field(..., description="Free disk space in GB", examples=[533.3])
    percent: float = Field(..., description="Disk usage percentage", examples=[44.1])
    read_speed_kbps: float = Field(..., description="Real-time disk read speed in KB/s", examples=[1250.0])
    write_speed_kbps: float = Field(..., description="Real-time disk write speed in KB/s", examples=[340.0])


class NetworkMetrics(BaseModel):
    download_speed_kbps: float = Field(..., description="Real-time network receive throughput in KB/s", examples=[240.5])
    upload_speed_kbps: float = Field(..., description="Real-time network transmit throughput in KB/s", examples=[45.2])
    total_recv_mb: float = Field(..., description="Total megabytes received since system boot", examples=[1420.8])
    total_sent_mb: float = Field(..., description="Total megabytes sent since system boot", examples=[310.4])


class SystemMetricsResponse(BaseModel):
    timestamp: str = Field(..., description="Timestamp of the metrics reading in ISO format")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    cpu: CPUMetrics = Field(..., description="CPU telemetry")
    memory: MemoryMetrics = Field(..., description="RAM and Swap telemetry")
    gpu: GPUMetrics = Field(..., description="GPU and CUDA telemetry")
    thermal: ThermalMetrics = Field(..., description="Thermal sensor temperatures")
    fan: FanMetrics = Field(..., description="Cooling fan metrics")
    power: PowerMetrics = Field(..., description="Power and voltage metrics")
    disks: List[DiskDeviceMetrics] = Field(..., description="Per-disk storage and real-time I/O throughput")
    network: NetworkMetrics = Field(..., description="Network throughput and totals")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service status (healthy or unhealthy)", examples=["healthy"])
    dependencies: Dict[str, dict] = Field(..., description="Status of infrastructure dependencies (CUDA, MinIO)")
