import numpy as np
from scipy.optimize import linprog


def generate(dim_x, dim_b, random_seed=0, num_x=0, vary_c=False, vary_a=False, vary_b=False, solve=False):
    """ Generate one or multiple linear problems (multiple if vary_b or vary_a is True) and num_x instances with an
    about even ratio of being within the LP polytope and not being within it. """
    if vary_c is False and vary_a is False and vary_b is False and dim_x == 2:
        # in this scenario, the following data generation functions works very well
        return generate_simple_lp_v2(dim_x, dim_b, random_seed, num_x, vary_c, vary_a, vary_b, solve)
    else:
        # otherwise, this data generation function should work well enough
        # there might be some unwanted dependencies though because data is generated in a way which makes the
        # resulting data set balanced (for the (non) feasible class)
        return generate_simple_lp(dim_x, dim_b, random_seed, num_x, vary_c, vary_a, vary_b, solve)


def generate_simple_lp_v2(dim_x, dim_b, random_seed, num_x, vary_c, vary_a, vary_b, solve):
    """ The goal of this function is to generate b independently from x, and only x dependent on A and b, so that the
    space of b can really be explored. This is done by generating c, A, and b first and independently and then
    generating x in such a way that the complete space of valid instances is part of the generation, the invalid
    instances are generated not too far from the decision boundary (the generated area is a rectangle) and the class
    ratio is about even. The current implementation does not support varying A or b. Unfortunately, the approach in
    this function, while it does work very well for 2 dimension (dim_x==2) does not generalize to larger dimensions.
    Data generation might still possible, but not with the same nice properties (like the whole space of valid instances
    being part of the data generation). It might not work at all for very large dimensions.
    It is recommended to use this function for dim_x==2, dim_b=False, and dim_a=False. Otherwise, it might not work very
    well or not work properly at all. """
    np.random.seed(random_seed)
    # chosen_cost_exception: see the print below
    chosen_cost_exception = dim_x == 2 and dim_b == 3 and not vary_a and not vary_b and not vary_c and random_seed == 7
    if chosen_cost_exception:
        print("For this case, the cost vector is chosen by hand. This should not mean any problem, because in theory "
              "this cost vector could have been randomly generated like that anyway. But since I like to use this seed "
              "for its constraints and its nice, understandable visualization, I also wanted to change the cost vector "
              "in a way which results in the optimal solution lying on the constraint intersection.")

    if num_x == 0:
        # if only the LP needs to be generated
        c = np.random.rand(dim_x)
        if chosen_cost_exception:
            c = np.array([0.5, 0.6])
        a = np.random.rand(dim_b, dim_x)
        # use dim_x as the upper bound for b
        b = np.random.rand(dim_b) * dim_x
        sol = 0
        if solve:
            sol = solve_lp(c, a, b)
        return c, a, b, np.zeros((1, dim_x)), np.array([[0]]), sol
    elif num_x < 10:
        raise ValueError("At least 10 instances need to be generated for some algorithms in this function to work "
                         "properly")

    # c can be generated randomly without considering the other values
    if not vary_c:
        # one c for all generated instances
        c = np.random.rand(dim_x)
        if chosen_cost_exception:
            c = np.array([0.5, 0.6])
    else:
        # one c for each instance
        c = np.random.rand(num_x, dim_x)

    sol = None

    if not vary_a:
        # one a for all generated instances
        a = np.random.rand(dim_b, dim_x)
        if not vary_b:
            # one b for all generated instances
            # use dim_x as the upper bound for b
            b = np.random.rand(dim_b) * dim_x

            # calculate the maximum value of each dimension which an instance could have and still be part of a valid
            # solution for the lp (for example if all other elements were 0)
            # if an element was bigger than that max value for the respective dimension, we would already know that
            # Ax <= b could not be true any more
            max_x_values = np.zeros(dim_x)
            # go over each constraint (one row in A and the corresponding element in b)
            for i in range(dim_b):
                max_values_this_dim = b[i] / a[i]
                if i == 0:
                    max_x_values = max_values_this_dim
                else:
                    # element wise minimum
                    max_x_values = np.minimum(max_x_values, max_values_this_dim)

            # now, generate values uniformly within the orthotope (hyperrectangle): this is will either always generate
            # more instances which are within the LP polytope or (on average) just as many instances within as without
            # the LP polytope (this is the case if the LP has only one constraint which is actually relevant, so for
            # example if dim_b=1 or if all but one constraint are always not violated if that one relevant constraint
            # is not violated) (this reasoning w applies as long as dim_x=2, not necessarily otherwise)

            x = np.random.rand(num_x, dim_x) * max_x_values
            y = label_x_feasible(x, a, b)

            # I decided, that having a class balance with at least 40% and the most 60% in favor of any of the two
            # classes is sufficient but otherwise, x is changed to make the classes more balanced
            # if the class ratio is not within that interval, the data is multiplied by some factor; this should make
            # sense since larger values can be useful to learn the task but values smaller than 0 (negative values) are
            # invalid inputs anyway, so we do not want to get those and "stretching" (multiplying) should make sense

            # it is not really easy to just calculate the "right" factor for the multiplication (for example, imagine,
            # albeit very unlikely, that all generated x are the same; then, no factor will be able to result in a
            # balanced data set; similar problems can also occur for more likely scenarios)
            # the best way I came up with, is using the current percentage as a guideline, for example if there are
            # currently 25% invalid instances, this number should be doubled, so we multiply by 2
            # so we use 0.5 (desired percentage) divided by current percentage of invalid instances

            # trying this nr_tries times, if that still does not result in data within the desired class ration, the
            # last data is still used but a warning is printed
            nr_tries = 50
            for i in range(nr_tries):
                # use a boundary, so that the correction factor does not become too large and we avoid dividing by 0
                # the boundary changes in a way which makes more extremer correction factors impossible for larger
                # iterations (a more "conservative" factor for later iterations)
                current_ratio = min(max(1-np.mean(y), 0.4*((i+1)/nr_tries)), 1-0.4*((i+1)/nr_tries))
                correction_factor = 0.5 / current_ratio
                x = x * correction_factor
                y = label_x_feasible(x, a, b)
                # print(i, np.mean(y))  # if you want to see how the class ratio changes
                if 0.4 <= np.mean(y) <= 0.6:
                    break
                if i == (nr_tries-1):
                    raise Warning("The data for the (non-) feasible task is not well balanced (class ratio:"
                                  f" {np.mean(y)} instances within the LP polytope")

            # solving the LP(s)
            if solve:
                if not vary_c:
                    sol = solve_lp(c, a, b)
                else:
                    sol = np.zeros((num_x, dim_x))
                    for i in range(num_x):
                        sol[i] = solve_lp(c[i], a, b)
        else:
            raise NotImplementedError("Not yet implemented")
    else:
        raise NotImplementedError("Not yet implemented")

    return c, a, b, x, y, sol


def generate_simple_lp(dim_x, dim_b, random_seed, num_x, vary_c, vary_a, vary_b, solve):
    """ Generate one or multiple linear problems (multiple if vary_b or vary_a is True) and num_x instances with an
    about even ratio of being within the LP polytope and not being within it. Results are either one-dimensional arrays
    of the respective length or have the shape (num_x, dim) with the respective dimension length 'dim'.
    Some performance improvements are certainly possible but at the time of writing not worth the extra effort. """
    np.random.seed(random_seed)

    if num_x == 0:
        # if only the LP needs to be generated
        c = np.random.rand(dim_x)
        a = np.random.rand(dim_b, dim_x)
        # by applying the following multiplication with 0.5 dim_x, it is ensured that on average Ax is about as large as
        # b if instances x would also be generated with values randomly within [0, 1]
        b = np.random.rand(dim_b) * (1 / 2 * dim_x)
        sol = 0
        if solve:
            sol = solve_lp(c, a, b)
        return c, a, b, np.zeros((1, dim_x)), np.array([[0]]), sol

    # c can be generated randomly without considering the other values
    if not vary_c:
        # one c for all generated instances
        c = np.random.rand(dim_x)
    else:
        # one c for each instance
        c = np.random.rand(num_x, dim_x)

    sol = None

    if not vary_a:
        # one a for all generated instances
        a = np.random.rand(dim_b, dim_x)
        if not vary_b:
            # one b for all generated instances

            # generate x
            x = np.random.rand(num_x, dim_x)

            # if for every comparison of dim_b ((Ax)[i] <= b[i] for all i in [1,...,dim_b]) the following probability
            # describes the probability of each of these comparisons being true, then the overall probability of
            # Ax <= b will be true in about 50% of all cases, this is used to try to generate instances which are
            # relatively balanced in terms of satisfying Ax <= b or not
            prob_per_b = 0.5 ** (1 / dim_b)

            # set b so that for every dimension prob_per_b instances result in smaller values
            b = np.zeros(dim_b)
            ax = np.matmul(a, x.T)  # dimensions: dim_b * num_x
            rest_ax = np.copy(ax)
            for i in range(b.shape[0]):
                b[i] = np.quantile(rest_ax[i], prob_per_b)
                rest_ax = rest_ax.T[rest_ax[i] <= b[i]]
                rest_ax = rest_ax.T

            # solving the LP(s)
            if solve:
                if not vary_c:
                    sol = solve_lp(c, a, b)
                else:
                    sol = np.zeros((num_x, dim_x))
                    for i in range(num_x):
                        sol[i] = solve_lp(c[i], a, b)
        else:
            # one b for each instance

            # generate x
            x = np.random.rand(num_x, dim_x)

            # set b so that Ax <= b in about half of the cases
            b = np.zeros((num_x, dim_b))
            for i in range(num_x):
                if np.random.choice([0, 1]):
                    # 50% of the time, just set b randomly, in the same interval as x
                    # especially for higher dimensions, this will mostly (but not always) results in Ax > b
                    b[i] = np.random.rand(dim_b)
                else:
                    # 50% of the time, set b so that it satisfies Ax <= b
                    # here, b is set randomly in the interval between Ax and a maximum value
                    # the maximum value is the largest possible result of Ax when A and x are both generated as values
                    # between 0 and 1 (which is the case)
                    ax = np.matmul(a, x[i].reshape((dim_x, 1)))
                    temp = np.full_like(ax, dim_x) - ax
                    b[i] = (ax + np.random.rand(dim_b).reshape((dim_b, 1)) * temp)[:, 0]

            # solving the LPs
            if solve:
                sol = np.zeros((num_x, dim_x))
                for i in range(num_x):
                    if not vary_c:
                        sol[i] = solve_lp(c, a, b[i])
                    else:
                        sol[i] = solve_lp(c[i], a, b[i])
    else:
        # one a for each instance
        if not vary_b:
            # one b for all generated instances

            # generate x
            x = np.random.rand(num_x, dim_x)

            # generate a
            a = np.random.rand(num_x, dim_b, dim_x)

            # if for every comparison of dim_b ((Ax)[i] <= b[i] for all i in [1,...,dim_b]) the following probability
            # describes the probability of each of these comparisons being true, then the overall probability of
            # Ax <= b will be true in about 50% of all cases, this is used to try to generate instances which are
            # relatively balanced in terms of satisfying Ax <= b or not
            prob_per_b = 0.5 ** (1 / dim_b)

            # set b so that for every dimension prob_per_b instances result in smaller values
            b = np.zeros(dim_b)
            ax = np.zeros((dim_b, num_x))  # dimensions: dim_b * num_x
            for i in range(num_x):
                ax[:, i] = np.matmul(a[i], x[i].T)
            rest_ax = np.copy(ax)
            for i in range(b.shape[0]):
                b[i] = np.quantile(rest_ax[i], prob_per_b)
                rest_ax = rest_ax.T[rest_ax[i] <= b[i]]
                rest_ax = rest_ax.T

            # solving the LPs
            if solve:
                sol = np.zeros((num_x, dim_x))
                for i in range(num_x):
                    if not vary_c:
                        sol[i] = solve_lp(c, a[i], b)
                    else:
                        sol[i] = solve_lp(c[i], a[i], b)
        else:
            # one b for each instance

            # generate a
            a = np.random.rand(num_x, dim_b, dim_x)

            # generate x
            x = np.random.rand(num_x, dim_x)

            # set b so that Ax <= b in about half of the cases
            b = np.zeros((num_x, dim_b))
            for i in range(num_x):
                if np.random.choice([0, 1]):
                    # 50% of the time, just set b randomly, in the same interval as x
                    # especially for higher dimensions, this will mostly (but not always) results in Ax > b
                    b[i] = np.random.rand(dim_b)
                else:
                    # 50% of the time, set b so that it satisfies Ax <= b
                    # here, b is set randomly in the interval between Ax and a maximum value
                    # the maximum value is the largest possible result of Ax when A and x are both generated as values
                    # between 0 and 1 (which is the case)
                    ax = np.matmul(a[i], x[i].reshape((dim_x, 1)))
                    temp = np.full_like(ax, dim_x) - ax
                    b[i] = (ax + np.random.rand(dim_b).reshape((dim_b, 1)) * temp)[:, 0]

            # solving the LPs
            if solve:
                sol = np.zeros((num_x, dim_x))
                for i in range(num_x):
                    if not vary_c:
                        sol[i] = solve_lp(c, a[i], b[i])
                    else:
                        sol[i] = solve_lp(c[i], a[i], b[i])

    y = label_x_feasible(x, a, b)

    return c, a, b, x, y, sol


def label_x_feasible(x, a, b):
    """ For all instances x, return a label describing whether x is a possible solution (not an optimal one) for the LP
    specified by a and b. Therefore, the label is 1 if x >= 0 and Ax <= b and 0 otherwise. """
    y = np.zeros((x.shape[0], 1))
    if len(a.shape) == 2 and len(b.shape) == 1:
        # fixed A and b
        for i in range(x.shape[0]):
            if np.all(x[i] >= 0) and np.all(np.matmul(a, x[i]) <= b):
                y[i] = 1
            else:
                y[i] = 0
    elif len(a.shape) == 2 and len(b.shape) == 2:
        # fixed A, varying b
        for i in range(x.shape[0]):
            if np.all(x[i] >= 0) and np.all(np.matmul(a, x[i]) <= b[i]):
                y[i] = 1
            else:
                y[i] = 0
    elif len(a.shape) == 3 and len(b.shape) == 1:
        # varying A, fixed b
        for i in range(x.shape[0]):
            if np.all(x[i] >= 0) and np.all(np.matmul(a[i], x[i]) <= b):
                y[i] = 1
            else:
                y[i] = 0
    else:
        # varying A and b
        for i in range(x.shape[0]):
            if np.all(x[i] >= 0) and np.all(np.matmul(a[i], x[i]) <= b[i]):
                y[i] = 1
            else:
                y[i] = 0
    return y


def solve_lp(c, a, b, method="interior-point"):
    """ Solve a given LP using scipy.linprog. Keep in mind that here, we maximize the cost (gain). """
    res = linprog(-c, a, b, method=method)
    return res.x
