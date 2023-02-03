import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import colors    # TODO remove? (if everything looks good, I can)
# from pathlib import Path  # TODO remove?

import linear_programs as lp


def visualize_loss(train_loss, test_loss, name):
    """ Plot the train and test loss of a neural network after learning. One graph shows the loss progress over all
    iterations, another one for only the last 10 iterations (can see whether it is still improving)."""
    nr_epochs = list(range(len(train_loss)+1))[1:]

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(nr_epochs, train_loss, label="Train")
    ax1.plot(nr_epochs, test_loss, label="Test")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_xticks(nr_epochs)
    ax1.set_xticklabels(str(epoch) for epoch in nr_epochs)
    ax1.set_title("Loss over all epochs")
    ax1.grid(True)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(nr_epochs[-10:], train_loss[-10:], label="Train")
    ax2.plot(nr_epochs[-10:], test_loss[-10:], label="Test")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_xticks(nr_epochs[-10:])
    ax2.set_xticklabels(str(epoch) for epoch in nr_epochs[-10:])
    ax2.set_title("Loss over all epochs")
    ax2.set_title("Loss over the last 10 epochs")
    ax2.grid(True)
    ax2.legend()

    plt.subplots_adjust(hspace=0.6)

    if name == "":
        plt.show()
    else:
        plt.savefig(f"models/Loss_{name}")
        plt.clf()
        plt.close()


def visualize_single_lps(identifier, attr, c=None, a=None, b=None, x=None, pred=None, name="", x_opt=None):
    """ Visualize the Single LP with the given attributions"""
    if identifier == "SingleLP":
        visualize_lp(attr, a, b, x, pred, name)
    elif identifier == "2D":
        visualize_2d(attr, c, a, b, x, pred, name, x_opt)
    elif identifier == "2Dx":
        visualize_2d_special(attr, a, b, x, name)
    else:
        raise ValueError(f"There is no visualization implemented for the \"{identifier}\" problem")


def visualize_lp(attr, a, b, x, pred, name):
    """ Visualize the attributions for a single LP as an equation Ax <= b. """

    # Get inputs in a more useful shape
    x = x.T
    attr = attr.T
    b = b.reshape((b.shape[0], 1))

    attribution_cmap = plt.cm.RdBu
    violation_cmap = plt.cm.Purples

    # idea: six subfigures (here separated by "|"): A | x | = | AX | <= | b
    fig, axs = plt.subplots(1, 6, figsize=(18, 3))

    # A
    draw_matrix(axs[0], a, "A")

    # x
    bounds = np.max(abs(attr))
    draw_matrix(axs[1], x, "x", color_values=attr, vmin=-bounds, vmax=bounds, cmap=attribution_cmap)

    # =
    draw_text(axs[2], "=")

    # Ax
    ax = np.matmul(a, x)
    diff = np.maximum(ax - b, np.zeros_like(b))
    draw_matrix(axs[3], ax, "Ax", color_values=diff, cmap=violation_cmap)

    # <=
    draw_text(axs[4], r"$\leq$")

    # b
    draw_matrix(axs[5], b, "b")

    set_colorbar(attribution_cmap, bounds)

    if np.all(np.matmul(a, x) <= b):
        label = 1
    else:
        label = 0
    fig.suptitle(f"Linear Problem, True: {label}, Predicted: {pred[0, 0]:.2f}", fontsize=30, ha="center", va="bottom")
    if name == "":
        plt.show()
    else:
        plt.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()


def set_colorbar(cmap, bounds, diverging=True, cax=None):
    """ Set the colormap according to the bounds (always normalized). """
    if diverging:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-bounds, vmax=bounds))
    else:
        raise Exception("fDGS") # TODO remove, then also cax
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=bounds))
    sm.set_array([])
    plt.colorbar(sm, cax=cax)


def draw_matrix(axs, matrix, name, color_values=None, vmin=None, vmax=None, cmap=plt.cm.Greens, subtitle=None):
    """ On the given subplot, draw the matrix. If color_values is given, color the matrix according to these values,
    otherwise the matrix is colored simply according to its values. Can also draw a vector (as amatrix where one
    dimension is 1). """
    if color_values is not None:
        # color the matrix according to the attribution
        im = axs.matshow(color_values, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        # color the matrix according to its values
        im = axs.matshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # set the text of the matrix fields
            if np.max(np.absolute(matrix)) >= 10:
                im.axes.text(j, i, str(int(matrix[i, j])), va="center", ha="center")
            else:
                im.axes.text(j, i, "{:.2f}".format(matrix[i, j]), va="center", ha="center")
    # remove ticks so it looks more like a matrix
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_title(name)
    if subtitle is not None:
        raise Exception("hfdjkds")  # TODO remove
        axs.set_xlabel(subtitle)


def draw_text(axs, text):
    """ On the given subplot, draw the text. """
    axs.text(0.5, 0.5, text, fontsize=50, va="center", ha="center")
    # remove axis so that only the text is visible
    axs.axis("off")


def visualize_2d(attr, c, a, b, x, pred, name, x_opt):  # TODO problem type only SingleLP (done I think)
    """ Visualizes the attribution of a 2-dimensional input instance on a 2-dimensional plot. Attribution for the input
    is shown in a vertical and horizontal line from that point where each line contains the attribution for one feature.
    The cost is also shown both in the background and the cost vector by an arrow on the top right. """  # TODO check caption
    # get inputs into a more useful form
    a = a.reshape(b.shape[0], c.shape[0])
    dim_b = b.shape[0]
    x = x.T
    b = b.reshape((b.shape[0], 1))
    x_opt = x_opt.T

    attr = attr.T

    color_bounds = 0
    color_bounds = max(color_bounds, np.max(np.abs(attr)))
    if color_bounds == 0:
        color_bounds += 0.1  # makes it so that all 0 attribution is white and not red
    attr_cmap = plt.cm.RdBu
    sm = plt.cm.ScalarMappable(cmap=attr_cmap, norm=plt.Normalize(vmin=-color_bounds, vmax=color_bounds))

    fig, ax = plt.subplots()

    # constraints
    x_max = 0
    y_max = 0
    for i in range(dim_b):
        x_values = [0, b[i, 0] / a[i, 0]]
        y_values = [b[i, 0] / a[i, 1], 0]
        x_max = max(x_values[1], x_max)
        y_max = max(y_values[0], y_max)
        ax.plot(x_values, y_values, c="gray", zorder=1)

    ax.scatter(x_opt[0], x_opt[1], c="g", zorder=3, marker="*", s=100)
    ax.scatter(x_opt[0], x_opt[1], zorder=0.5, s=1000, facecolors="none", edgecolors="g", lw=1.5)

    # adjust x_max and y_max which will be the axis bounds
    if round(x_max) < x_max:
        x_max = round(x_max) + 1
    else:
        x_max = round(x_max)

    if round(y_max) < y_max:
        y_max = round(y_max) + 1
    else:
        y_max = round(y_max)

    # for some edge cases, we cut off the image if a constraint line is much too steep or flat
    # this way, we always get images with a decent ratio
    if x_max * 1.5 < y_max:
        y_max = x_max * 1.5
    elif y_max * 1.5 < x_max:
        x_max = y_max * 1.5

    # ...or just fix the bounds, this makes sense as long as we look at lps with most values generated within [0, 1] or
    # if we know what the bounds of the plot should be
    # x_max = 2
    # y_max = 2

    # x
    if x is not None:
        ax.scatter(x[0], x[1], c="black", zorder=4)
    else:
        raise Exception("fdsds")  # todo remove this and remove if else construct

    # attribution for x (it is shown starting from its axis (so the x[0] attribution is the line straight up))
    ax.plot([x[0], x[0]], [y_max, 0], c=sm.to_rgba(attr[0, 0]), zorder=2, lw=2)
    ax.plot([0, x_max], [x[1], x[1]], c=sm.to_rgba(attr[1, 0]), zorder=2, lw=2)

    # background / cost (gain) vector
    nr_steps = 10**3
    if x_max > y_max:
        x_steps = nr_steps
        one_x_step = x_max / nr_steps
        y_steps = round(nr_steps * y_max / x_max)
        if y_steps < nr_steps * y_max / x_max:
            y_steps += 1
        one_y_step = y_max / y_steps
    else:
        y_steps = nr_steps
        one_y_step = y_max / nr_steps
        x_steps = round(nr_steps * x_max / y_max)
        if x_steps < nr_steps * x_max / y_max:
            x_steps += 1
        one_x_step = x_max / x_steps
    background_array = np.zeros((y_steps, x_steps))
    for i in range(background_array.shape[0]):
        for j in range(background_array.shape[1]):
            x_here = np.array([(j+0.5)*one_y_step, (i+0.5)*one_x_step])
            background_array[i, j] = np.matmul(c.T, x_here)
    ax.imshow(background_array, cmap=plt.cm.cool, origin="lower", alpha=0.5, extent=(0, x_max, 0, y_max),
              interpolation="bilinear")

    # arrow for the cost vector, go towards the middle of the first reached axis (so there is enough space for the rest)
    if c[0] > c[1]:
        dx = -x_max/2
        dy = - c[1] * ((x_max + dx) / c[0])
    else:
        dy = -y_max/2
        dx = - c[0] * ((y_max + dy) / c[1])
    ax.arrow(x_max, y_max, dx, dy, fc="magenta", width=max(x_max, y_max)*0.01, length_includes_head=True)

    ax.set_xlim([0, x_max])
    ax.set_ylim([0, y_max])

    sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin=0, vmax=np.max(background_array)))
    sm.set_array([])
    plt.colorbar(sm)

    # attribution colormap
    set_colorbar(attr_cmap, color_bounds)

    # TODO: other encodings
    if np.all(np.matmul(a, x) <= b):
        label = 1
    else:
        label = 0
    plt.title(f"True: {label}, Predicted: {pred[0, 0]:.2f}", fontsize=10, ha="center", va="bottom")

    if name == "":
        plt.show()
    else:
        plt.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()


def visualize_2d_special(attr, a, b, x, name):
    """ Visualization for a 2-dimensional LP where there is attribution for points over the whole input space (of course
    only with a certain precision). So you can see which points get which attribution.
    There are three plots. The top one shows the attribution sum, and the other two plots show the attribution on the
    input values (dimensions) individually."""
    # get inputs into a more useful form
    b = b.reshape((b.shape[0], 1))
    attr = attr.reshape((np.unique(x[:, 0]).shape[0], np.unique(x[:, 1]).shape[0], 2))
    attr = np.transpose(attr, (1, 0, 2))

    color_bounds = np.max(np.abs(attr))
    color_bounds_sum = np.max(np.abs(attr[:, :, 0] + attr[:, :, 1]))
    attr_cmap = plt.cm.RdBu
    sm = plt.cm.ScalarMappable(cmap=attr_cmap, norm=plt.Normalize(vmin=-color_bounds, vmax=color_bounds))
    sm_sum = plt.cm.ScalarMappable(cmap=attr_cmap, norm=plt.Normalize(vmin=-color_bounds_sum, vmax=color_bounds_sum))

    fig2, fig3 = None, None
    if name == "":  # this means we want to show, so having 3 subplots makes sense (can see them at the same time)
        fig, ax = plt.subplots(nrows=3)
    else:  # for saving the figures, it is more useful to have them separately (more flexibility)
        fig, axis = plt.subplots()
        fig2, axis2 = plt.subplots()
        fig3, axis3 = plt.subplots()
        ax = [axis, axis2, axis3]

    # constraints
    for i in range(b.shape[0]):
        x_values = [0, b[i, 0] / a[i, 0]]
        y_values = [b[i, 0] / a[i, 1], 0]
        for axis in ax:
            axis.plot(x_values, y_values, c="gray", zorder=1)
            axis.tick_params(axis="both", which="major", labelsize=12)

    # image bounds
    x_max = x.max(axis=0)[0]
    y_max = x.max(axis=0)[1]

    # I think while bilinear interpolation looks better, no interpolation is more informative (true to the data)
    # attribution sum
    ax[0].imshow(attr[:, :, 0] + attr[:, :, 1], cmap=attr_cmap, origin="lower", alpha=0.5, extent=(0, x_max, 0, y_max),
                 vmin=-color_bounds_sum, vmax=color_bounds_sum)  # , interpolation="bilinear")
    # horizontal (left <-> right) feature
    ax[1].imshow(attr[:, :, 0], cmap=attr_cmap, origin="lower", alpha=0.5, extent=(0, x_max, 0, y_max),
                 vmin=-color_bounds, vmax=color_bounds)  # , interpolation="bilinear")
    # vertical (down <-> up) feature
    ax[2].imshow(attr[:, :, 1], cmap=attr_cmap, origin="lower", alpha=0.5, extent=(0, x_max, 0, y_max),
                 vmin=-color_bounds, vmax=color_bounds)  # , interpolation="bilinear")

    sm.set_array([])
    for i, axis in enumerate(ax):
        # have larger graph than image bounds so you can really see when the data stops
        axis.set_xlim([0, x_max*1.1])
        axis.set_ylim([0, y_max*1.1])
        if i == 0:
            fig.colorbar(sm_sum, ax=axis)
        else:
            fig.colorbar(sm, ax=axis)

    ax[0].set_title("Attribution Sum", fontsize=20, ha="center", va="bottom")
    ax[1].set_title("Attribution Horizontal Feature", fontsize=20, ha="center", va="bottom")
    ax[2].set_title("Attribution Vertical Feature", fontsize=20, ha="center", va="bottom")
    folder = "2D_special"
    if name == "":
        plt.show()
    else:
        fig.savefig(f"figures/{folder}/{name}_0_sum.pdf", bbox_inches="tight")
        fig2.savefig(f"figures/{folder}/{name}_1_horizontal.pdf", bbox_inches="tight")
        fig3.savefig(f"figures/{folder}/{name}_2_vertical.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()


def visualize_part_of_lp(attr, a, b, x, x_indices, b_indices, pred, name):
    """ Visualize the attributions for a single LP as an equation Ax <= b but only showing the features given by
    x_indices and the constraints given by b_indices. All calculations use the full LP (for Ax). """

    # Get inputs in a more useful shape
    x = x.T
    attr = attr.T
    b = b.reshape((b.shape[0], 1))

    attribution_cmap = plt.cm.RdBu
    violation_cmap = plt.cm.Purples

    # idea: six subfigures (here separated by "|"): A | x | --> | AX | <= | b
    fig, axs = plt.subplots(1, 6, figsize=(18, 3))

    # A
    v_min_max = np.max(np.absolute(a[b_indices][:, x_indices]))
    draw_matrix(axs[0], a[b_indices][:, x_indices], "A", cmap=plt.cm.PiYG, vmin=-v_min_max, vmax=v_min_max)

    # x
    bounds = np.max(abs(attr))
    draw_matrix(axs[1], x[x_indices], "x", color_values=attr, vmin=-bounds, vmax=bounds, cmap=attribution_cmap)

    # =
    draw_text(axs[2], r"$\rightarrow$")

    # Ax
    ax = np.matmul(a, x)
    diff = np.maximum(ax - b, np.zeros_like(b))[b_indices]
    draw_matrix(axs[3], ax[b_indices], "Ax", color_values=diff, vmin=0, cmap=violation_cmap)

    # <=
    draw_text(axs[4], r"$\leq$")

    # b
    draw_matrix(axs[5], b[b_indices], "b")

    set_colorbar(attribution_cmap, bounds)

    if np.all(np.matmul(a, x) <= b):
        label = 1
    else:
        label = 0
    fig.suptitle(f"Linear Problem, True: {label}, Predicted: {pred[0, 0]:.2f}\nFeatures: {x_indices}, Constraints: {b_indices}", fontsize=30, ha="center", va="bottom")
    if name == "":
        plt.show()
    else:
        plt.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()


def visualize_large_lp(attr, a, b, x, pred, name, small_x_dim, small_b_dim, visualize, use_largest_vio):
    """ Given attribution for an LP with a large number of dimensions, visualize the LP and attribution with the matrix/
     vector visualization (as in visualize_lp) but before that selecting only a number of input features (x, see
     small_x_dim), and a number of constraints (small_b_dim) so that the visualization remains compact. The features of
     x are chosen in such a way that the features with the largest absolute attributions are visualized. The constraints
     are chosen either with respect to their largest violation (use_largest_vio) or such that the constraints which are
     closest to being equal to Ax are shown (use_largest_vio=False). Also returns the indices for the two absolute
     largest attributions (can be used for a specific 2D visualization). """
    print(pred[0][0])

    # get indices for small_x_dim largest attributions
    largest_attr_inds = np.argpartition(np.abs(attr[0]), -small_x_dim)[-small_x_dim:]
    # activate the following lines if you also want to show the smallest attributions
    # smallest_attr_inds = np.argpartition(np.abs(attr[0]), small_x_dim)[:small_x_dim]
    # largest_attr_inds = np.concatenate((smallest_attr_inds, largest_attr_inds), axis=0)

    # Ax - b
    diff_ax_b = np.matmul(a, x.T) - b.reshape((b.shape[0], 1))
    diff_ax_b = diff_ax_b.reshape(diff_ax_b.shape[0])

    if use_largest_vio:
        # indices for largest constraint violations
        b_criterion = np.argpartition(diff_ax_b, -small_b_dim)[-small_b_dim:]
    else:
        # indices for constraint closest to fulfilling Ax=b
        b_criterion = np.argpartition(np.abs(diff_ax_b), small_b_dim)[:small_b_dim]

    if visualize:
        # do the visualization showing the fewer dimensions
        visualize_part_of_lp(attr[:, largest_attr_inds], a, b, x, largest_attr_inds, b_criterion, pred, name)

    # calculate and return the indices for the two largest absolute attributions
    largest_two_attr_inds = np.argpartition(np.abs(attr[0]), -2)[-2:]
    return largest_two_attr_inds
