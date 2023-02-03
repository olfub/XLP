import SingleLP

###################################################################
# train and save models
###################################################################
if __name__ == "__main__":
    # overwrite_argv = "--epochs 2 --num-x 100000 --seed 7 --problem 2D --save-model --save-name Feasibility".split(" ")  # TODO combine this and lower code? (done and works I think)
    # SingleLP.main(overwrite_argv)

    encodings = ["Feasibility", "CostZero", "CostMinusone", "CostCostpenalty", "DistanceMin", "DistanceMinAbs", "OptimalEdges"]  # TODO mention new and dont forget to remove unused ones completely
    for enc in encodings:
        overwrite_argv = f"--epochs 20 --num-x 100000 --seed 7 --problem 2D --save-model --enc {enc} --save-name {enc}"
        SingleLP.main(overwrite_argv.split(" "))

    overwrite_argv = f"--epochs 25 --num-x 100000 --save-model --save-name five_dims"
    SingleLP.main(overwrite_argv.split(" "))
