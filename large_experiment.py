import SingleLP

###################################################################
# large experiment
###################################################################
if __name__ == "__main__":
    overwrite_argv = "--epochs 40 --num-x 1000000 --dim-x 10000 --dim-b 30 --seed 0 --enc Feasibility --save-model " \
                     "--save-name large_model --no-data-storing --architecture 11-8192-cons --batch-size 256 " \
                     "--test-batch-size 1024"
    SingleLP.main(overwrite_argv.split(" "))
