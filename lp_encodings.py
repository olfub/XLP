import numpy as np


def change_feasible_encoding(enc, c, a, b, x, y):
    """ This function can be used on top of other generation functions which generate data for the (not) feasible task,
    so the task where an instance either fulfills Ax <= b or does not. This function takes this data (with y containing
    values with either 1 (Ax <= b) or 0 (else), and changes y, thereby changing the encoding. The respective
    possibilities are explained in the respective functions. """
    if enc == "CostZero":
        y = cost_zero(c, x, y)
    elif enc == "CostMinusone":
        y = cost_minusone(c, x, y)
    elif enc == "CostCostpenalty":
        y = cost_costpenalty(c, a, b, x, y)
    elif enc == "DistanceMin":
        y = distance_min(a, b, x, y)
    elif enc == "DistanceMinAbs":
        y = distance_min_abs(a, b, x, y)
    elif enc == "OptimalEdges":
        y = optimal_edges_special_example(a, b, x, y)
    return y


def cost_zero(c, x, y):
    """ Here, the cost/gain is used for all valid instances while the invalid instances remain at 0. This approach is
    similar to the simple (not) feasible task but it contains more variation/information on the valid instances by
    including the cost vector in that calculation. """
    for i in range(y.shape[0]):
        if y[i] == 1:
            y[i] = np.matmul(c, x[i])
    return y


def cost_minusone(c, x, y):
    """ Here, the cost/gain is used for all valid instances while the invalid instances get assigned -1. This approach
    is similar to cost_zero but it also ensures that all invalid instances have a worse score than any valid instance
    where the score is always nonnegative. """
    for i in range(y.shape[0]):
        if y[i] == 1:
            y[i] = np.matmul(c, x[i])
        else:
            y[i] = -1
    return y


def cost_costpenalty(c, a, b, x, y):
    """ Here, the cost/gain is used for all valid instances and all invalid instances get assigned a cost/gain of a
    close (ideally the closest, but this often is approximated for a reduced complexity) point, with a penalty applied
    on this to make sure that being in the invalid area is never the optimal solution. Still, this way small violations
    are not punished as extremely, as with other encodings. """
    for i in range(y.shape[0]):
        if y[i] == 1:
            y[i] = np.matmul(c, x[i])
        else:
            # idea: the point is invalid, check every single constraint (row in A with one value in b), identify the
            # constraint with the largest violation and move to the closest point where this constraint is not violated
            # any more, now check every other single constraint, ... , repeat until no more constraint violation (this
            # way, we do not get the closest point in the feasible area, but we get an estimation of this point and we
            # know that this estimation is either exact or further away than the actual closest feasible point)
            # now calculate the distance of this new and the original point (euclidean distance), use the cost from the
            # valid point but reduce it depending on the distance (further away --> reduce more)
            # x_valid: the (approximately) closest valid point (at this point not calculated yet)
            x_valid = np.copy(x[i])
            # account for constraints until x_valid is valid
            acceptable_error = 1e-15
            while np.min(b - np.matmul(a, x_valid)) < -acceptable_error:
                most_violated_constraint = np.argmin(b - np.matmul(a, x_valid))
                # trying to solve x[i] - t*a[most_violated_constraint] = x_b (so that x_b is on the constraint, i.e.
                # a[most_violated_constraint]x_b = b[most_violated_constraint]) with those two equations, you can
                # calculate t:
                t = sum(a[most_violated_constraint][j] * x_valid[j] for j in range(len(a[most_violated_constraint])))
                t = t - b[most_violated_constraint]
                t = t / sum(a[most_violated_constraint][j] ** 2 for j in range(len(a[most_violated_constraint])))
                # now we can get the new point with x[i] - t*a[most_violated_constraint]
                x_valid = x_valid - t * a[most_violated_constraint]
                # however, it is possible for this point to contain a negative coordinate
                if np.any(x_valid < 0):
                    # in 2 dimension, there is a unique closest point on the axis we can calculate here
                    if x_valid.shape[0] == 2:
                        # will want to set that to 0
                        most_violated_feature = np.argmin(x_valid)
                        x_valid[most_violated_feature] = 0
                        # will set most_violated_feature to 0, calculate the other one (f2: the other feature)
                        f2 = 1 - most_violated_feature
                        x_valid[f2] = b[most_violated_constraint] / a[most_violated_constraint][f2]
                    # in more than 2 dimensions, multiple possibilities exist
                    # I think it would make sense to choose the point (out of those possibilities) which is closest to
                    # the original point but this results in a not as simple calculation
                    else:
                        raise NotImplementedError("Implementation still missing")
            # now we have a close (not always the closest, but a good estimation) valid point
            dist = np.linalg.norm(x[i] - x_valid)
            # reference distance (from 0 to the valid point), this is used to apply the penalty to the invalid point:
            # the penalty scales linearly, with 100% equaling the reference distance, for example: if the distance of
            # the input point and x_valid is 30% of the reference distance, 30% of the gain of x_valid is subtracted
            ref_dist = np.linalg.norm(x_valid)
            penalty = min(1, dist/ref_dist)
            # use the valid cost
            cost = np.matmul(c, x_valid)
            # label with the valid cost but penalized linearly by the distance
            y[i] = cost * (1-penalty)
    return y


def distance_min(a, b, x, y):
    """ Here, it is calculated how far a point is away from the boundary. This is done by calculating Ax and comparing
    this to b (b - Ax). If the result has no negative elements, we know the instance is valid. On the other hand, any
    negative values stand for constraint violations. We use the smallest value (either the strongest constraint
    violation or (if there are none) the constraint which is closest to being violated) as a measure for how far the
    instance is from the boundary. """
    for i in range(y.shape[0]):
        y[i] = min(b - np.matmul(a, x[i]))
    return y


def distance_min_abs(a, b, x, y):
    """ Here, it is calculated how far a point is away from the boundary. This is done by calculating Ax and comparing
    this to b (b - Ax). If the result has no negative elements, we know the instance is valid. On the other hand, any
    negative values stand for constraint violations. We use the smallest value (either the strongest constraint
    violation or (if there are none) the constraint which is closest to being violated) as a measure for how far the
    instance is from the boundary. In this function, we use the absolute values, so we do not distinguish between valid
    and invalid instances. This approach might be more useful if you want to "find" the boundary, because the boundary
    has the smallest values (a smaller value is closer to the boundary). """
    for i in range(y.shape[0]):
        y[i] = abs(min(b - np.matmul(a, x[i])))
    return y


def optimal_edges_special_example(a, b, x, y):
    """ Contrary to the previous encoding implementations, this implementation of the Optimal Edges encoding does not
    work on general data but instead only on the one shown in the paper. It is not difficult to implement this
    encoding for any specific LP but a general implementation was not possible here due to time constraints.
    For this specific LP, this function implements the Optimal Edges encoding, i.e. the distance of any point to its
    closest vertex (here without including the origin). """
    # the next few lines make sure this method is not used for the wrong seed (or at least highly unlikely)
    a_comp = np.array([[0.4384, 0.7234], [0.9779, 0.5385], [0.5011, 0.0720]])
    b_comp = np.array([0.5368, 0.9997, 1.3584])
    correct_values = np.all(np.isclose(a, a_comp, rtol=0.001)) and np.all(np.isclose(b, b_comp, rtol=0.001))
    if not correct_values:
        raise ValueError("This method only works for one specific seed and set of data (seed: 7, x_dim:2, b_dim:3, both"
                         " c, a, and p constant).")
    axis_0 = np.array([min(b / a[:, 0]), 0])
    axis_1 = np.array([0, min(b / a[:, 1])])
    # also need the intersection of the first two constraint (we know that, for this specific example)
    # so Ax = b has to hold for both, it follows:
    x_1 = (b[0] - ((a[0, 0] * b[1]) / a[1, 0])) / (-((a[0, 0] * a[1, 1]) / a[1, 0]) + a[0, 1])
    x_0 = (b[1] - (a[1, 1] * x_1)) / a[1, 0]
    intersection = np.array([x_0, x_1])
    points = [axis_0, axis_1, intersection]
    for i in range(y.shape[0]):
        distances = [np.linalg.norm(point - x[i]) for point in points]
        y[i] = min(distances)
    return y
