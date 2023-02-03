import SingleLP

###################################################################
# small test experiment
###################################################################
if __name__ == "__main__":
    overwrite_argv = "--epochs 3 --num-x 10000 --seed 0 --problem 2D --dim-x 10 --dim-b 3 --enc Feasibility --save-model --save-name SmallTest --architecture 10-8192-dec"
    # overwrite_argv += " --vis --vis-special --vis-special-steps 50 --vis-attr sal --vis-attr-opt 0-1"
    SingleLP.main(overwrite_argv.split(" "))
