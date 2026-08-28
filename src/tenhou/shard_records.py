# -*- coding: utf-8 -*-
"""Record sharding (BEFORE tensorization): split the 237 day record files
into uniform record shards (65536 records each) with real progress lines."""
import gzip, glob, os, time

SRC = "C:/agentwork/data/processed/tenhou/records-*.jsonl.gz"
OUT = "C:/agentwork/data/processed/tenhou/record_shards"
SHARD_SIZE = 65536

def flush(lines, idx):
    path = os.path.join(OUT, "records_shard-%06d.jsonl.gz" % idx)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fo:
        fo.write("\n".join(lines) + "\n")
    os.replace(tmp, path)

def main():
    files = sorted(glob.glob(SRC))
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    n = 0
    shards = 0
    done = 0
    buf = []
    for fi, f in enumerate(files, 1):
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    n += 1
                    buf.append(line)
                    if len(buf) >= SHARD_SIZE:
                        flush(buf, shards)
                        shards += 1
                        buf = []
        except Exception as e:
            print("skip", f, str(e)[:60], flush=True)
        done = fi
        if done % 5 == 0 or done == len(files):
            dt = time.time() - t0
            print("progress shard file=%d/%d records=%d shards=%d rate=%.0f/s elapsed=%ds"
                  % (done, len(files), n, shards, n / max(dt, 1e-6), dt), flush=True)
    if buf:
        flush(buf, shards)
        shards += 1
    dt = time.time() - t0
    print("DONE shard files=%d records=%d shards=%d rate=%.0f/s elapsed=%ds"
          % (len(files), n, shards, n / max(dt, 1e-6), dt), flush=True)

if __name__ == "__main__":
    main()