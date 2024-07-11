import numpy as np

def read_txt(file_name):
    with open(file_name) as f:
        info = f.readline()
        info = info.split("\t")
        nr_constraints = int(info[0])
        dim_size = int(info[1])
        nr_support_constraints = 2*dim_size + 1
        nr_random_constraints = nr_constraints - nr_support_constraints
        a = np.zeros((nr_random_constraints, dim_size))
        b = np.zeros(nr_random_constraints)
        c = np.zeros(dim_size)
        for i in range(nr_support_constraints):
            # skip support constraints
            f.readline()
        for i in range(nr_random_constraints):
            constraints_str = f.readline()
            constraint_elems = constraints_str.split("\t")
            a[i, :] = [int(a_elem) for a_elem in constraint_elems[:-1]]
            b[i] = int(constraint_elems[-1])
        cost_str = f.readline()
        c[:] = [int(cost_elem) for cost_elem in cost_str.split("\t")[:dim_size]]
    return c, a, b

def check_feasibles(a, x, b):
    ax = np.matmul(a, x.T)
    feasible = 0
    infeasible = 0
    feas_indices = []
    print_count = 0
    for i in range(ax.shape[1]):
        if np.all(ax[:, i] < b):
            feasible += 1
            feas_indices.append(i)
        else:
            infeasible += 1
        if i/ax.shape[1] >= print_count:
            print(print_count)
            print_count += 0.01
    feas_ratio = feasible/(feasible+infeasible)
    return feas_ratio, feas_indices

def create_x(c, a, b, alpha, nr_samples, closeness=0.01, seed=0):
    if nr_samples == 0:
        return c, a, b, None, None, None

    # Generate three batches of instances
    # 1. feasible instances
    # 2. infeasible instances
    # 3. infeasible instances near the decision boundary
    nr_samples_per_area = nr_samples // 3
    np.random.seed(seed)
    print("Step 1")

    # 1. feasible instances
    # assuming, that randomly generating instances in the region bounded by alpha "often enough" returns
    # a feasible instance to be computationally reasonable
    feas_count = 0
    feasibles = []
    while feas_count < nr_samples_per_area:
        x = np.random.rand(nr_samples_per_area, a.shape[1]) * alpha
        feas_ratio, feas_indices = check_feasibles(a, x, b)
        feasibles.append(x[feas_indices])
        feas_count += len(feas_indices)
        print(feas_count/nr_samples_per_area)
    # as many feasible instances as we want (nr_samples_per_area)
    x_feasible = np.concatenate(feasibles, axis=0)[:nr_samples_per_area]  # should usually be enough, otherwise (nr_samples_per_area > x_feasible2.shape[0]) TODO
    print("finished1")

    # 2. infeasible instances
    # get infeasible instances from previously generated batch, should be more than enough
    x = np.random.rand(nr_samples_per_area*2, a.shape[1]) * alpha  # generating two times should be enough (generating more infeasible instances anyway)
    feas_ratio, feas_indices = check_feasibles(a, x, b)
    infeas_indices = [index for index in range(nr_samples_per_area*2) if index not in feas_indices]
    x_infeasible = x[infeas_indices[:nr_samples_per_area]]  # if it does not work (nr_samples_per_area > len(infeas_indices)), TODO
    print("finished2")

    # 3. closer infeasible instances
    # the idea is, that usually too little feasible instances are generated
    # by now generating more in 1., the feasible area is much denser with generated instances
    # to make up for that, more infeasible instances are generated
    # these however, are generated in a way where they are close the feasible area
    # this should (hopefully) help the model since these points should be more information
    # than points further out
    # an instances is "close" if it turns feasible when all features are moved some percent closer to
    # the point in the middle (which is always feasible, per the problem generation approach)
    # overall, the least dense region should be infeasible instance not close to the decision boundary
    # this should be fine and those easily recognizable as infeasible
    middle_point = np.full((a.shape[1]), alpha/2)
    x = np.random.rand(nr_samples_per_area*2, a.shape[1]) * alpha  # generating two times should be enough (generating more infeasible instances anyway)
    feas_ratio, feas_indices = check_feasibles(a, x, b)
    infeas_indices = [index for index in range(nr_samples_per_area*2) if index not in feas_indices]
    # remove feasible instances
    x = x[infeas_indices]
    feas_ratio, feas_indices = check_feasibles(a, x, b)
    assert feas_ratio == 0

    # store the desired close instances in x_close
    x_close = np.zeros((x.shape[0], a.shape[1]))
    filled = 0

    while x.shape[0] > 0:
        # move instances closer to the point in the middle (which we know is feasible)
        x_new = x.copy()
        x_new -= middle_point
        x_new *= 1-closeness
        x_new += middle_point
        # check which are feasible now
        feas_ratio, feas_indices = check_feasibles(a, x_new, b)
        infeas_indices = [index for index in range(x.shape[0]) if index not in feas_indices]
        if len(feas_indices) > 0:
            # store the infeasible instances (of x) which are close to being feasible (in x_new) in x_close
            x_close[filled:filled+len(feas_indices)] = x[feas_indices]
            filled += len(feas_indices)
            # continue only with instances which are still infeasible
            x = np.zeros((x.shape[0]-len(feas_indices), a.shape[1]))
            x = x_new[infeas_indices]
        else:
            # no new feasible instances, continue with instances closer to the middle point
            x = x_new
        print(filled)

    # shuffle to make sure any kind of instances (coming from any region) can be selected
    np.random.shuffle(x_close)
    x_close = x_close[:nr_samples_per_area]
    assert feas_ratio == 1

    # add feasibility labels
    x_feasible = np.concatenate((x_feasible, np.full((x_feasible.shape[0], 1), 1)), axis=1)
    x_infeasible = np.concatenate((x_infeasible, np.full((x_infeasible.shape[0], 1), 0)), axis=1)
    x_close = np.concatenate((x_close, np.full((x_close.shape[0], 1), 0)), axis=1)

    # combine to one dataset and shuffle
    x_all = np.concatenate((x_feasible, x_infeasible, x_close), axis=0)
    np.random.shuffle(x_all)

    # separate data and label again
    x = x_all[:, :-1]
    y = x_all[:, -1]
    print("finished3")

    return c, a, b, x, y, None

# c, a, b = read_txt("lps/10000_30.txt")
# c, a, b, x, y, sol = create_x(c, a, b, 100, 1000000, seed=0)
# np.save("lps/10000_30_c.npy", c)
# np.save("lps/10000_30_a.npy", a)
# np.save("lps/10000_30_b.npy", b)
# np.save("lps/10000_30_x.npy", x)
# np.save("lps/10000_30_y.npy", y)

# c, a, b, x, y, sol = create_x(c, a, b, 100, 1000000, seed=1)
# np.save("lps/10000_30_c_1.npy", c)
# np.save("lps/10000_30_a_1.npy", a)
# np.save("lps/10000_30_b_1.npy", b)
# np.save("lps/10000_30_x_1.npy", x)
# np.save("lps/10000_30_y_1.npy", y)