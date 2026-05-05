import ray
import time

class FusionProgress:
    def __init__(self, grid, fuse_task_remote, bb_min, per_view_transforms, output_path, overlap_strategy):
        self.grid = grid
        self.fuse_task_remote = fuse_task_remote
        self.bb_min = bb_min 
        self.per_view_transforms = per_view_transforms
        self.output_path = output_path
        self.overlap_strategy = overlap_strategy
        self.prefix = "[fusion]"

    def run_with_progress(self):
        futures = []
        total_cells = len(self.grid)
        completed = 0
        failed = 0
        t_run0 = time.perf_counter()
        last_pct_printed = -1

        print(f"{self.prefix} submitting {total_cells} tasks", flush=True)

        for grid_block in self.grid:
            futures.append(self.fuse_task_remote.remote(grid_block, self.bb_min, self.per_view_transforms, self.output_path, self.overlap_strategy))

            done, futures = ray.wait(futures, num_returns=1, timeout=0)
            while done:
                try:
                    ray.get(done[0])
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"{self.prefix}[ERROR] task failed: {type(e).__name__}: {e}", flush=True)

                pct_int = int((completed / max(total_cells, 1)) * 100.0)
                if pct_int > last_pct_printed:
                    last_pct_printed = pct_int
                    elapsed = time.perf_counter() - t_run0
                    rate = completed / max(elapsed, 1e-9)
                    eta_s = (total_cells - completed) / max(rate, 1e-9)
                    print(
                        f"{self.prefix} Progress: ok={completed - failed} failed={failed} total={total_cells} ({pct_int}%) "
                        f"elapsed={elapsed/60:.1f}m rate={rate:.2f} cells/s eta={eta_s/60:.1f}m",
                        flush=True,
                    )

                done, futures = ray.wait(futures, num_returns=1, timeout=0)

        while futures:
            done, futures = ray.wait(futures, num_returns=1, timeout=1.0)
            if not done:
                continue

            try:
                ray.get(done[0])
                completed += 1
            except Exception as e:
                failed += 1
                print(f"{self.prefix}[ERROR] task failed: {type(e).__name__}: {e}", flush=True)

            pct_int = int((completed / max(total_cells, 1)) * 100.0)
            if pct_int > last_pct_printed:
                last_pct_printed = pct_int
                elapsed = time.perf_counter() - t_run0
                rate = completed / max(elapsed, 1e-9)
                eta_s = (total_cells - completed) / max(rate, 1e-9)
                print(
                    f"{self.prefix} Progress: ok={completed - failed} failed={failed} total={total_cells} ({pct_int}%) "
                    f"elapsed={elapsed/60:.1f}m rate={rate:.2f} cells/s eta={eta_s/60:.1f}m",
                    flush=True,
                )
    
    def run(self):
        self.run_with_progress()