"""
Wrapper launcher that fixes the libuv TCPStore issue on Windows.

Usage:
    python launch.py --nproc_per_node=8 dataloader.py
"""
import torch.distributed as _dist

_OrigTCPStore = _dist.TCPStore

class _PatchedTCPStore(_OrigTCPStore):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_libuv", False)
        super().__init__(*args, **kwargs)

_dist.TCPStore = _PatchedTCPStore

import torch.distributed.elastic.rendezvous.static_tcp_rendezvous as _rdzv
_rdzv.TCPStore = _PatchedTCPStore

from torch.distributed.run import main
main()
