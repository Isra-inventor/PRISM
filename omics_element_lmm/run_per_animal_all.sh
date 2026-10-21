#!/bin/bash
# per-animal analyses for every element (Cu already done)
for el in Na Mg P K Ca Fe Zn Ni Ni_resFe; do
  python3 per_animal_assoc.py --data-dir /tmp/claude-0/-home-user-PRISM/d8faa7bd-080b-523b-a2ac-e3bce5dd5c4a/scratchpad/new --out-dir results_v2 --element $el > results_v2/per_animal/log_${el}.txt 2>&1
  python3 per_animal_k_of_5.py --data-dir /tmp/claude-0/-home-user-PRISM/d8faa7bd-080b-523b-a2ac-e3bce5dd5c4a/scratchpad/new --out-dir results_v2 --element $el --alpha 0.10 >> results_v2/per_animal/log_${el}.txt 2>&1
done
echo ALLDONE > results_v2/per_animal/ALLDONE
