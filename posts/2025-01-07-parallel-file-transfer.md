---
aliases:
- /parallel-transfer/2025/01/07/parallel-file-transfer
categories:
- ssh
- scp
- performance
- terminal
date: '2025-01-07'
description: Transferring large projects with thousands of small files over SSH can be painfully slow. Here's how we solved it with parallel transfers.
layout: post
title: Parallel File Transfer Over SSH When You Have Thousands of Files
toc: true

---

# The Problem

I recently needed to transfer a Screen Studio project (about 3.5GB) from my MacBook to a remote machine. Screen Studio projects are bundles containing thousands of small video segments. When I tried the obvious `scp -r` command, it was painfully slow.

```bash
scp -r project.screenstudio user@remote:~/Downloads/
```

This was taking forever because SCP transfers files sequentially, and with thousands of small files, the latency overhead for each file transfer adds up significantly.

# First Attempt: Parallel Rsync

I thought about using `rsync` with parallel transfers. My initial approach was to split the files into batches and run multiple rsync processes simultaneously:

```bash
#!/bin/bash
SRC="/path/to/project.screenstudio"
DEST="user@remote:~/Downloads/project.screenstudio"

# Create destination directory structure
ssh user@remote "mkdir -p ~/Downloads/project.screenstudio/recording"

# Split files into 4 parallel batches
cd "$SRC/recording"

transfer_batch() {
    local start=$1
    local end=$2
    local batch_num=$3
    ls -1 channel-1-display-0-*.m4s | sed -n "${start},${end}p" | \
    rsync -avz --files-from=- . "user@remote:~/Downloads/project.screenstudio/recording/"
}

TOTAL=2686
BATCH_SIZE=$((TOTAL / 4))

# Run 4 parallel batches
transfer_batch 1 $BATCH_SIZE 1 &
transfer_batch $((BATCH_SIZE + 1)) $((BATCH_SIZE * 2)) 2 &
transfer_batch $((BATCH_SIZE * 2 + 1)) $((BATCH_SIZE * 3)) 3 &
transfer_batch $((BATCH_SIZE * 3 + 1)) $TOTAL 4 &

wait
```

# The Issue with Parallel Batches

The parallel script seemed like a good idea, but when I checked the destination, I found:
- Source: 3.5GB, 5388 files
- Destination: 3.3GB, 4573 files
- **Missing: ~815 files!**

The parallel transfers had some issues and didn't complete all files. This is a common problem with parallel file transfers - race conditions, connection issues, or process management can cause incomplete transfers.

# The Simple Solution: Just Use Rsync

After the parallel approach partially failed, I went back to basics. A simple rsync command completed the remaining files:

```bash
rsync -avz --progress /path/to/project.screenstudio/ user@remote:~/Downloads/project.screenstudio/
```

Rsync is smart about:
- Only transferring changed files (incremental transfers)
- Compressing data during transfer (`-z` flag)
- Preserving file permissions and timestamps (`-a` flag)
- Showing progress (`--progress` flag)
- Resuming interrupted transfers

# Key Takeaways

1. **Start simple**: A single rsync is often sufficient and more reliable than complex parallel solutions
2. **Check your transfers**: Always verify file counts and sizes after transfer
3. **Monitor progress**: Use `--progress` with rsync or monitor with:
   ```bash
   watch -n 2 'ssh user@remote "du -sh ~/Downloads/project && ls ~/Downloads/project/recording/ | wc -l"'
   ```
   (On macOS, install watch with `brew install watch` first)

4. **Parallel isn't always better**: For file transfers, reliability often beats raw speed

# Monitoring Transfer Progress

During the transfer, I found it helpful to monitor both the size and file count:

```bash
# Check current progress
ssh user@remote "du -sh ~/Downloads/project.screenstudio"
ssh user@remote "ls ~/Downloads/project.screenstudio/recording/ | wc -l"
```

# Final Result

After using rsync to complete the transfer:
- **Source**: 3.5GB, 5388 files
- **Destination**: 3.5GB, 5388 files

Perfect match, transfer complete!

# Bonus: Why Screen Studio Files Are Like This

Screen Studio projects contain thousands of small `.m4s` files (media segments). These are typically:
- Video segments from screen recording (channel-1-display-0-*.m4s)
- Audio segments from microphone (channel-2-microphone-0-*.m4s)
- Metadata files (project.json, meta.json, etc.)

This structure allows for efficient editing and export within Screen Studio, but makes file transfers slower due to the sheer number of files.

# Alternative Approaches

If you frequently need to transfer large projects with many files, consider:
- **Compressing first**: Create a tar/zip archive, transfer the single file, then extract
- **Using a network share**: Mount the remote directory via SSHFS or similar
- **rsync with more parallelism**: Tools like `parallel` or GNU parallel can help, but add complexity

For most cases, simple `rsync -avz --progress` is your best bet.
