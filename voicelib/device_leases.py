"""Compatibility names for the single canonical kilix-device-lease component.

Providers written against voicelib.device_leases use the same implementation,
namespace, grant and quarantine history as current Voice's accelerator client.
No lease implementation or alternate registry lives in this package.
"""
from pathlib import Path
import kilix_device_lease as _home

if _home.VERSION != 'kilix.device-lease/v1':
    raise ImportError('canonical device lease interface differs')
if Path(_home.__file__).resolve().is_relative_to(Path(__file__).resolve().parent.parent):
    raise ImportError('device lease implementation must be outside the Voice package')

VERSION = _home.VERSION
LeaseError = _home.LeaseError
QueueStatus = _home.QueueStatus
Lease = _home.Lease
acquire = _home.acquire
WORKLOADS = _home.WORKLOADS
MAX_WAIT_SECONDS = _home.MAX_WAIT_SECONDS
MAX_QUEUE = _home.MAX_QUEUE
MAX_WORKLOAD_QUEUE = _home.MAX_WORKLOAD_QUEUE
