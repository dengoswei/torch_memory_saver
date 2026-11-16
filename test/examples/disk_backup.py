import logging
import fcntl
import os
import shutil
import sys
import tempfile

import torch

from torch_memory_saver import torch_memory_saver


def run(hook_mode: str):
    # Small chunk (before init) so the tensors below span many chunks.
    os.environ["TMS_DISK_BACKUP_CHUNK_MB"] = "1"
    torch_memory_saver.hook_mode = hook_mode
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    disk_dir = tempfile.mkdtemp(prefix="tms_disk_backup_", dir=os.environ.get("TMS_TEST_DISK_DIR"))
    print(f"disk_backup_dir={disk_dir}")
    node = int(os.environ.get("TMS_TEST_NUMA_NODE", "0"))
    torch_memory_saver.configure_direct_disk_backup(disk_dir, numa_node=node)
    torch_memory_saver.configure_direct_disk_backup(disk_dir, numa_node=node)
    try:
        torch_memory_saver.configure_direct_disk_backup(disk_dir, numa_node=node + 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("conflicting process-global placement accepted")
    info = torch_memory_saver.get_disk_backup_info()
    assert info["direct"] and info["observed_numa_nodes"] == [node]
    assert info["staging_address"] % info["memory_alignment"] == 0
    assert info["staging_bytes"] == 1024 * 1024
    assert info["fd_flags"] & os.O_DIRECT
    assert info["fd_descriptor_flags"] & fcntl.FD_CLOEXEC
    print(f"native_disk_backup_info={info}")
    try:
        _run(disk_dir)
    finally:
        shutil.rmtree(disk_dir, ignore_errors=True)


def _run(disk_dir: str):
    chunk_bytes = 1 * 1024 * 1024
    numel = chunk_bytes * 13 // 4 + 12345  # ~13.4 fp32 chunks: multi-chunk, non-even tail

    print("Allocate tensor_with_disk_backup")
    with torch_memory_saver.region(enable_disk_backup=True):
        big = torch.randn((numel,), dtype=torch.float32, device='cuda')
        small = torch.full((7,), 10, dtype=torch.uint8, device='cuda')
    big_oracle = big.detach().to('cpu').clone()
    small_oracle = small.detach().to('cpu').clone()

    print("Allocate tensor_without_backup")
    with torch_memory_saver.region(enable_disk_backup=False):
        no_backup = torch.full((20_000_000,), 20, dtype=torch.uint8, device='cuda')

    assert torch.equal(big.cpu(), big_oracle)
    assert no_backup[:3].tolist() == [20, 20, 20]

    disk_sizes = set()
    initial = torch_memory_saver.get_disk_backup_info()
    logical_total = None
    for _ in range(3):
        previous = torch_memory_saver.get_disk_backup_info()
        torch_memory_saver.pause()

        files = os.listdir(disk_dir)
        total = sum(os.path.getsize(os.path.join(disk_dir, f)) for f in files)
        assert len(files) >= 2, f"expected a backup file per disk-backed alloc, got {files}"
        assert total >= big.numel() * 4, f"backup smaller than the paused tensor: {total}"
        disk_sizes.add(total)
        info = torch_memory_saver.get_disk_backup_info()
        # MemPool allocations can share tensors and contain allocator slack.
        actual_logical = info["logical_write_bytes"] - previous["logical_write_bytes"]
        assert actual_logical >= big.numel() * 4 + small.numel()
        if logical_total is None:
            logical_total = actual_logical
        assert actual_logical == logical_total
        assert info["fd_flags"] & os.O_DIRECT
        assert info["fd_descriptor_flags"] & fcntl.FD_CLOEXEC
        assert info["padded_write_bytes"] >= info["logical_write_bytes"]

        occupy = torch.full((20_000_000,), 30, dtype=torch.uint8, device='cuda')
        del occupy
        torch.cuda.empty_cache()

        torch_memory_saver.resume()

        assert torch.equal(big.cpu(), big_oracle)
        assert torch.equal(small.cpu(), small_oracle)
        assert no_backup[:3].tolist() != [20, 20, 20]

    info = torch_memory_saver.get_disk_backup_info()
    assert info["logical_write_bytes"] - initial["logical_write_bytes"] == 3 * logical_total
    assert info["logical_read_bytes"] - initial["logical_read_bytes"] == 3 * logical_total
    assert info["padded_write_bytes"] - initial["padded_write_bytes"] == 3 * total
    assert info["padded_read_bytes"] - initial["padded_read_bytes"] == 3 * total
    assert len(disk_sizes) == 1, f"disk usage grew across cycles: {sorted(disk_sizes)}"
    print(f"OK: {big.numel()} fp32 elems restored bit-identically across 3 cycles, "
          f"disk constant at {next(iter(disk_sizes))} bytes")


if __name__ == '__main__':
    run(hook_mode=sys.argv[1])
