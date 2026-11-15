#!/bin/bash
# Progress overview. Safe to run on the login node at any time.
cd "$(dirname "$0")/.."
echo "== Jobs in queue =="
squeue -u "$USER" -o "%.14i %.18j %.9T %.10M %.12l %R" 2>/dev/null || echo "(squeue not available)"
echo
echo "== Finished pieces =="
echo "E1:          $(ls results/e1/*/results.json 2>/dev/null | wc -l) run(s)"
echo "Validation:  $(python3 -c "import json;print('PASSED' if json.load(open('results/validation/validation.json'))['passed'] else 'FAILED')" 2>/dev/null || echo pending)"
echo "Ablation:    $(ls -d results/ablation/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null | cut -d_ -f1-2 | sed 's/_20[0-9]*//' | sort -u | tr '\n' ' ')"
echo "Convergence: $(ls results/convergence/*/results.json 2>/dev/null | wc -l) run(s)"
echo "CV folds:    $(ls results/cv/folds/*.json 2>/dev/null | wc -l) / 120"
echo "Seeds:       $(ls results/seeds/*/results.json 2>/dev/null | wc -l) / 10"
echo
echo "== Errors in logs (last 5) =="
grep -l -E "Traceback|Error|CANCELLED|oom|Killed" logs/*.out 2>/dev/null | tail -5 || true
echo
source .venv/bin/activate && python src/08_report.py --results_root results \
    --reference_json reference/multitask_results_20251206_115420.json --logs_dir logs | tail -40
