electrode-kit HPC submission templates
======================================

Files
-----
1) electrode_kit_job.env
   - Centralized job parameters.
   - Fill PROJECT_ROOT / DATASET_ROOT / TRAINING_OUTPUT_ROOT first.

2) submit_electrode_kit.sh
   - Generic shell submission script (recommended when your cluster submits .sh files).
   - Submit by your platform command, for example:
     sbatch submit_electrode_kit.sh
     qsub submit_electrode_kit.sh

3) submit_electrode_kit.slurm
   - Submit on Slurm cluster:
     sbatch submit_electrode_kit.slurm

4) submit_electrode_kit.pbs
   - Submit on PBS/Torque cluster:
     qsub submit_electrode_kit.pbs

5) cpu_stage_localtmp.slurm
   - CPU-only stage with node-local output root ($SLURM_TMPDIR), then rsync back to /scratch.
   - Useful when shared filesystem small-file I/O is the bottleneck.

Why these templates avoid interaction
-------------------------------------
electrode-kit.py only enters interactive prompts when --dataset-root is missing.
These templates always pass --dataset-root and all major args explicitly,
then run with "</dev/null" to ensure non-interactive batch behavior.

Common usage modes
------------------
1) Incremental training from existing dataset
   TRAIN_COUNT=0
   VERIFY_COUNT=0
   TEST_COUNT=0
   RECOMPUTE_EXISTING_FACTORS=0

2) Add new structures then continue full pipeline
   TRAIN_COUNT/VERIFY_COUNT/TEST_COUNT > 0
   RECOMPUTE_EXISTING_FACTORS=0 (recommended)

3) Force full factor recomputation
   RECOMPUTE_EXISTING_FACTORS=1

Notes
-----
- Existing factor files are skipped by default in current electrode-kit.
- Test split writes both JSON and XLSX structure-factor outputs.
- Keep --num-workers consistent with allocated CPU cores.
