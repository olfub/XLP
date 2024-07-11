import warnings
import pickle
import numpy as np

import linear_programs as lp


def get_data_single_lp(args, num_x=None, random_seed=None):
    """ Load data for a single LP neural network as specified by the problem argument. """

    num_x = args.num_x if num_x is None else num_x
    seed = args.seed if random_seed is None else random_seed

    if args.dim_x == 10000 and args.dim_b == 30:
        print("Getting data for large experiment (10000, 30)")
        try:
            c = np.load("lps/10000_30_c.npy")
            a = np.load("lps/10000_30_a.npy")
            b = np.load("lps/10000_30_b.npy")
            x = np.load("lps/10000_30_x.npy")
            y = np.load("lps/10000_30_y.npy")
            x1 = np.load("lps/10000_30_x_1.npy")
            y1 = np.load("lps/10000_30_y_1.npy")
            x = np.concatenate((x, x1), axis=0)
            y = np.concatenate((y, y1), axis=0)
            print(x.shape, y.shape)
            sol = None
            print("Data loaded")
        except:
            print("Generating...")
            from linear_programs_plus import read_txt, create_x
            c, a, b = read_txt("lps/10000_30.txt")
            c, a, b, x, y, sol = create_x(c, a, b, 100, num_x)
            np.save("lps/10000_30_c.npy", c)
            np.save("lps/10000_30_a.npy", a)
            np.save("lps/10000_30_b.npy", b)
            np.save("lps/10000_30_x.npy", x)
            np.save("lps/10000_30_y.npy", y)
            print("Data generated")
        y = np.reshape(y, (y.shape[0], 1))
    elif args.problem == "SingleLP":
        print("Generate SingleLP")
        c, a, b, x, y, sol = lp.generate(args.dim_x, args.dim_b, seed, num_x, vary_c=False, vary_a=False,
                                         vary_b=False, solve=False)
    elif args.problem == "2D":
        print("Generate 2D")
        c, a, b, x, y, sol = lp.generate(2, args.dim_b, seed, num_x, vary_c=False, vary_a=False,
                                         vary_b=False, solve=True)
    else:
        raise ValueError(f"There is no implementation for problem \"{args.problem}\".")
    if args.dim_x != x.shape[1]:
        warn_str = "Warning, the specified dimension dim_x: {} does not fit the dimension of the problem {} which is " \
                   "{}. The problem dimension will be used.".format(args.dim_x, args.problem, x.shape[1])
        warnings.warn(warn_str)
    return c, a, b, x, y, sol


def get_data_paramlp(args, param, num_x=None, random_seed=None, **kwargs):
    """ Load data for a ParamLP neural network as specified by the problem argument. """

    num_x = args.num_x if num_x is None else num_x
    seed = args.seed if random_seed is None else random_seed

    if param == "b":
        vary_a = False
        vary_b = True
    elif param == "a":
        vary_a = True
        vary_b = False
    else:
        raise ValueError("Currently, param must be either \"a\" or \"b\"")

    if args.problem == "ParamLP":
        c, a, b, x, y, sol = lp.generate(args.dim_x, args.dim_b, seed, num_x, vary_b=vary_b, vary_a=vary_a, solve=True)
    else:
        raise RuntimeError(f"Problem {args.problem} is unknown for generating data")
    if args.dim_x != x.shape[1]:
        warn_str = "Warning, the specified dimension dim_x: {} does not fit the dimension of the toy problem {} " \
                   "which is {}. The toy problem dimension will be used.".format(args.dim_x, args.problem, x.shape[1])
        warnings.warn(warn_str)
    return c, a, b, x, y, sol


# The problem: I want to separate the model training and the visualization, so I want to visualize instance which fit
# the trained model (same constant c, a, b; depending on the model) and generate data in the same way as done for
# training and testing (looking at other, more different data is a different problem, this is not about that). What made
# this a little difficult is that I can not just use the same random seed and generate less instances (because x is
# randomly generated before all constraints) but I do not want to always generate as many instances as used for training
# because, in some situations, this might take enough time to be annoying. This problem is solved by storing the
# information corresponding to the model. While I think "better" solutions for this problem should exist, this is what I
# came up with and it works sufficiently well, therefore it is used.
class InfoObject:
    """ Stores information about the save model, like LP parameter values (constraints) and generated data. """
    def __init__(self, problem, name):
        self.problem = problem
        self.name = name
        self.args = None
        self.c = None
        self.a = None
        self.b = None
        self.x = None
        self.y = None
        self.sol = None

    def set_data(self, args, **kwargs):
        self.args = args
        self.c = kwargs["c"] if "c" in kwargs else None
        self.a = kwargs["a"] if "a" in kwargs else None
        self.b = kwargs["b"] if "b" in kwargs else None
        self.x = kwargs["x"] if "x" in kwargs else None
        self.y = kwargs["y"] if "y" in kwargs else None
        self.sol = kwargs["sol"] if "sol" in kwargs else None

    def get_args(self):
        return self.args

    def get_data(self):
        # some attribution method use numpy.random methods, this makes sure that the args seed also applies to that
        np.random.seed(self.args.seed)
        return self.c, self.a, self.b, self.x, self.y, self.sol


def make_info(args, **kwargs):
    info = InfoObject(args.problem, args.save_name)
    info.set_data(args, **kwargs)
    return info


def store_info(path, args, **kwargs):
    info = make_info(args, **kwargs)
    with open(f"{path}{args.save_name}.pickle", "wb") as file:
        pickle.dump(info, file)
    return info


def load_info(path, name):
    with open(f"{path}{name}.pickle", "rb") as file:
        info = pickle.load(file)
    return info
