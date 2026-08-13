# PROJECT_STATE.md
Last updated: 13 Aug 2026 (mid-session, 3x3 DQN just completed)

## VERIFIED Environment
- macOS, MacBook Air M4, 16GB RAM
- SUMO 1.27.1, SUMO_HOME set correctly
- conda env `thesis`, Python 3.11.15
- Key packages: Stable-Baselines3 2.9.0, PyTorch 2.13.0, NumPy 2.4.6, Gymnasium 1.3.0

## VERIFIED Architecture
- env/traffic_env.py: TrafficSignalEnv (Gymnasium-style multi-agent env wrapping TraCI)
  - Auto-discovers TLS intersections + neighbours from any .net.xml via discover_tls_and_neighbours() (sumolib-based)
  - 6-feature local+neighbour observation per agent, fixed size regardless of network size
  - Discrete(2) action per agent: hold / switch phase, min green = 10s enforced
  - Reward: r = -alpha*delta_waiting - beta*queue + delta*throughput - lambda*switch_cost
    (alpha=0.3, beta=0.2, delta=0.1, lambda=0.5)
  - throughput = vehicles that left an intersection's approach lanes since last step (FIXED bug, was vehicle-presence count before)
- env/train_dqn_marl.py: independent DQN per agent, manual replay buffer + manual polyak_update target sync (FIXED bug, target net was never updated before)
- env/train_ppo_marl.py: independent PPO per agent, manual rollout buffer + GAE
- env/collect_baseline.py: fixed-time baseline runner, logs same metrics as RL eval
- env/evaluate_trained.py: loads trained models, deterministic eval, logs same metrics
- env/analyze_results.py: descriptive stats + paired t-test/Wilcoxon + comparison plots
- env/validate_phase1.py: gate-check script (model files exist, result files exist, no dupes, tripinfo cross-check)
- All training/eval/baseline scripts take --grid argument (2x2/3x3/4x4/5x5), paths follow
  network/gridNxN/, routes/gridNxN/, models/gridNxN/, results/gridNxN/ convention

## VERIFIED Completed Work
- Phase 1 (2x2, bug-fixed): full retrain + evaluation (60 runs) + analysis + validation ALL PASSED
  - Corrected results: low/medium/dynamic show strong significant improvement (waiting -50% to -70%, p<0.001)
  - High demand now shows near-null/negative results with corrected code (PPO travel time +10.9% worse,
    not significant p=0.07; PPO throughput significantly worse -14.3%, p=0.044) -- REAL finding,
    different from the pre-fix thesis draft's (overstated) high-demand numbers
- Phase 2 refactor: traffic_env.py + all 4 dependent scripts generalized and regression-tested on grid2x2
- 3x3 grid: network built (9 TLS), demand recalibrated (medium, period=1.0, matches 2x2's congestion severity)
- 3x3 DQN medium: TRAINING COMPLETE, 9 models saved, training log saved

## VERIFIED Commands
cd ~/thesis_project/env
python train_dqn_marl.py --grid GRID --scenario SCENARIO --total_steps 50000
python train_ppo_marl.py --grid GRID --scenario SCENARIO --total_steps 50000
python collect_baseline.py --grid GRID --scenario SCENARIO --seed SEED
python evaluate_trained.py --grid GRID --controller dqn_or_ppo --scenario SCENARIO --seed SEED
python analyze_results.py
python validate_phase1.py

## NOT YET DONE / PENDING
- 3x3 PPO training: NOT STARTED
- 3x3 evaluation (fixed-time + DQN + PPO, seeds 11-15): NOT STARTED
- 3x3 result analysis/validation: NOT STARTED
- 4x4, 5x5 grids: NOT STARTED
- Scalability cross-grid analysis/plots: NOT STARTED
- Master results file, scalability_summary.csv: NOT STARTED
- README.md, requirements.txt: NOT STARTED
- Thesis .docx: STILL CONTAINS OLD PRE-BUGFIX CHAPTER 4 NUMBERS. Not yet updated with:
  corrected 2x2 results, any 3x3+ results, front matter, TOC regeneration,
  RQ count alignment, queue-length results, statistical rigor additions,
  reference deduplication, wording fixes (scalable claim, generalised->adapted,
  unseen demand->stochastic replications, action-space description, Table 4.3 typo,
  Section 2.x placeholders, Ch2 ride-hailing opening, past-perfect tense cleanup)

## Known Issues / Risks
- Demand scaling across grid sizes must be calibrated by congestion severity (waiting time), not raw
  vehicle count -- naive proportional scaling caused gridlock on first 3x3 attempt (144 teleports)
- Reward magnitude changed after the throughput bug fix (now smaller/often negative) -- expected, not a bug
- Backup of pre-Phase-1 project exists at ~/thesis_project_backup_20260813_0343

## Design Decisions
- True independent learners (no shared policy/parameters) throughout, per original thesis proposal
- Grid sizes tested incrementally: 2x2 -> 3x3 -> 4x4 -> 5x5, stop and validate after each
- Scalability experiment scope: medium demand only, Fixed-Time/DQN/PPO, seeds 11-15
- Never discard failed/negative results -- report honestly even if scalability degrades
