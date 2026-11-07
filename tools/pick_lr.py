"""Pick the SGD learning rate of one grid cell of the optimizer study: the learning rate with the lowest best-epoch validation MSE
(the 'Vali Loss' of the training log, i.e. the H_max validation loss that early stopping monitors), seed 2023.
Usage: python tools/pick_lr.py <out_file> <log path template with {lr}> lr1 lr2 ...
A missing log or a log without a finite validation loss counts as inf; ties go to the first learning rate.
Writes the chosen learning rate (one token, as given) to out_file and prints "log,lr,best_vali_mse" per learning rate
(scripts/run_optimizer_study.sh appends these lines to grid_vali.csv / gridown_vali.csv). Exit 1 if no learning rate is finite."""
import math
import re
import sys

out, tmpl, lrs = sys.argv[1], sys.argv[2], sys.argv[3:]
best = None
for lr in lrs:
    v = math.inf
    try:
        vals = [float(x) for x in re.findall(r'Vali Loss: ([-+0-9.eEnaif]+)', open(tmpl.format(lr=lr)).read())]
        vals = [x for x in vals if not math.isnan(x)]
        if vals:
            v = min(vals)
    except FileNotFoundError:
        pass
    print(f'{tmpl.format(lr=lr).split("/")[-1]},{lr},{v:.7f}')
    if best is None or v < best[1]:
        best = (lr, v)
if math.isinf(best[1]):
    sys.exit('no finite validation MSE for any learning rate: ' + tmpl)
open(out, 'w').write(best[0])
