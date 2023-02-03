import SingleLP

###################################################################
# create figures
###################################################################
if __name__ == "__main__":
    overwrite_argv = f"--epochs 25 --num-x 100000 --load-model --save-name five_dims --vis --vis-attr ig-sal-fp-lime --num-vis 1 --vis-next 2 --vis-save"
    SingleLP.main(overwrite_argv.split(" "))

    overwrite_argv = f"--epochs 25 --num-x 100000 --load-model --save-name five_dims --vis --vis-attr ig-sal-fp-lime --num-vis 1 --vis-next 78 --vis-save"
    SingleLP.main(overwrite_argv.split(" "))

    methods = ["func", "ig", "gshap", "sal", "fp", "lime"]
    method_opt = {"func": ["0"], "ig": ["1-0-0"], "gshap": ["1-2-0"], "sal": ["0-0"], "fp": ["10-0.3-0", "10-0.1-0"], "lime": ["0.3-0", "0.1-0"]}

    overwrite_argv_base = "--epochs 20 --num-x 100000 --seed 7 --problem 2D --load-model --vis --vis-special --vis-save "

    encodings = ["Feasibility", "CostZero", "CostMinusone", "CostCostpenalty", "DistanceMin", "DistanceMinAbs", "OptimalEdges"]
    for enc in encodings:
        overwrite_argv_enc = overwrite_argv_base + f"--enc {enc} --save-name {enc} "
        for method in methods:
            for opt in method_opt[method]:
                if enc == "Feasibility":
                    opt = opt[:-1] + "1"
                overwrite_argv = overwrite_argv_enc + f"--vis-attr {method} --vis-attr-opt {opt}"
                SingleLP.main(overwrite_argv.split(" "))