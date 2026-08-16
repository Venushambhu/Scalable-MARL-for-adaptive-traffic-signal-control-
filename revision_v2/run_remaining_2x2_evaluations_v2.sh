#!/bin/bash
set -e

echo "============================================================"
echo "2x2 LEARNED-CONTROLLER FINAL EVALUATIONS"
echo "============================================================"

for scenario in low high dynamic
do
    echo
    echo "============================================================"
    echo "DQN | 2x2 | $scenario | seeds 11-15"
    echo "============================================================"

    python "evaluate_dqn_2x2_${scenario}_v2.py" \
        | tee "dqn_v2_2x2_${scenario}_evaluation_output.txt"

    echo
    echo "============================================================"
    echo "PPO | 2x2 | $scenario | seeds 11-15"
    echo "============================================================"

    python "evaluate_ppo_2x2_${scenario}_v2.py" \
        | tee "ppo_v2_2x2_${scenario}_evaluation_output.txt"
done

echo
echo "============================================================"
echo "PASS: ALL REMAINING 2x2 LEARNED EVALUATIONS FINISHED"
echo "============================================================"
