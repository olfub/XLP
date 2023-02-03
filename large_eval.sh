#!/bin/sh

python3 ./SingleLP.py --epochs 40 --num-x 1000000 --dim-x 10000 --dim-b 30 --seed 0 --enc Feasibility --load-model --save-name large_model --no-data-storing --architecture 11-8192-cons --batch-size 256 --test-batch-size 1024 --vis --vis-large --vis-special-steps 50 --vis-next 3 --vis-attr ig --vis-attr-opt 1-0-1 --vis-save --vis-large-options 5-5-1-0
wait
python3 ./SingleLP.py --epochs 40 --num-x 1000000 --dim-x 10000 --dim-b 30 --seed 0 --enc Feasibility --load-model --save-name large_model --no-data-storing --architecture 11-8192-cons --batch-size 256 --test-batch-size 1024 --vis --vis-large --vis-special-steps 50 --vis-next 3 --vis-attr sal --vis-attr-opt 0-1 --vis-save --vis-large-options 5-5-1-0
wait
python3 ./SingleLP.py --epochs 40 --num-x 1000000 --dim-x 10000 --dim-b 30 --seed 0 --enc Feasibility --load-model --save-name large_model --no-data-storing --architecture 11-8192-cons --batch-size 256 --test-batch-size 1024 --vis --vis-large --vis-special-steps 50 --vis-next 3 --vis-attr lime --vis-attr-opt 10-1 --vis-save --vis-large-options 5-5-1-0
wait
python3 ./SingleLP.py --epochs 40 --num-x 1000000 --dim-x 10000 --dim-b 30 --seed 0 --enc Feasibility --load-model --save-name large_model --no-data-storing --architecture 11-8192-cons --batch-size 256 --test-batch-size 1024 --vis --vis-large --vis-special-steps 50 --vis-next 3 --vis-attr fp --vis-attr-opt 50-10-1 --vis-save --vis-large-options 5-5-1-0
wait

echo "Everything is finished"