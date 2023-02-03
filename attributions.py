import warnings

import torch
import captum.attr as capttr
from captum._utils.models.linear_model import SkLearnLinearModel

# identifiers for all implemented attribution methods
ALL_ATTR_METHODS = ["ig", "gshap", "sal", "fp", "lime"]


def get_options(attr_method, option_str):
    """ Takes an identifier input (option_str) which can easily be given as a parameter when running a file and returns
    a tuple which contains the options for the respective attribution method. Those options still have to actually be
    applied by the functions using them (this is not done in this function). """
    option_elems = option_str.split("-")
    if attr_method == "func":
        if option_str == "":
            # default setting
            return (True,)
        # use sigmoid function (default/predict) or without sigmoid (not-predict)
        sigmoid = True if option_elems[0] == "1" else False
        return (sigmoid,)
    elif attr_method == "ig":
        if option_str == "":
            # default setting
            return True, 0, True
        # multiply_by_inputs
        multiply_by_inputs = True if option_elems[0] == "1" else False
        # baseline: 0 --> zero baseline, 1 --> "edges" baseline
        baseline = int(option_elems[1])
        # use sigmoid function (default/predict) or without sigmoid (not-predict)
        sigmoid = True if option_elems[2] == "1" else False
        return multiply_by_inputs, baseline, sigmoid
    elif attr_method == "gshap":
        if option_str == "":
            # default setting
            return True, 2, True
        # multiply_by_inputs
        multiply_by_inputs = True if option_elems[0] == "1" else False
        # baseline: 0 --> zero baselines (with noise), 1 --> "edges" baseline (with noise), 2 --> data baseline
        # (data baseline means random baseline in the data space, this should be equivalent to choosing random points
        # which are actually part of the training data)
        baseline = int(option_elems[1])
        # use sigmoid function (default/predict) or without sigmoid (not-predict)
        sigmoid = True if option_elems[2] == "1" else False
        return multiply_by_inputs, baseline, sigmoid
    elif attr_method == "sal":
        if option_str == "":
            # default setting
            return False, True
        # absolute gradients
        abs_grad = True if option_elems[0] == "1" else False
        # use sigmoid function (default/predict) or without sigmoid (not-predict)
        sigmoid = True if option_elems[1] == "1" else False
        return abs_grad, sigmoid
    elif attr_method == "fp":
        if option_str == "":
            # default setting
            return 50, 0.1, True
        # how often to calculate attribution with new random near instance
        nr_iterations = int(option_elems[0])
        # how large the hypercube (e.g. 2 dimensions: square) should be where perturbed instances can appear
        pert_range = float(option_elems[1])
        # use sigmoid function (default/predict) or without sigmoid (not-predict)
        sigmoid = True if option_elems[2] == "1" else False
        return nr_iterations, pert_range, sigmoid
    elif attr_method == "lime":
        if option_str == "":
            # default setting
            return 0.1, True

        # options for LIME where currently only one possibility is implemented (see captum, LIME documentation for more
        # details): similarity function, to_interp function, model

        # how large the hypercube (e.g. for 2 dimensions: square) should be where perturbed instances can appear
        pert_range = float(option_elems[0])
        # use sigmoid function (default/predict) or without sigmoid (not-predict)
        sigmoid = True if option_elems[1] == "1" else False
        return pert_range, sigmoid
    else:
        raise ValueError(f"Attribution method {attr_method} does not exist.")


def parse_methods(args):
    """ Parse the string from the file parameter to the list of attribution methods. Also print a warning if a specified
    attribution methods does not exist. """
    attr_str = args.vis_attr.lower()
    all_attr_methods = ALL_ATTR_METHODS
    if attr_str == "all":
        attr_methods = all_attr_methods
    else:
        attr_methods = []
        for attr_method in attr_str.split("-"):
            # because func is not really an "attribution method" but still should be treated similarly
            if attr_method in all_attr_methods or attr_method == "func":
                attr_methods.append(attr_method)
            else:
                warn_str = f"There is no implementation for the attribution method \"{attr_method}\"."
                warnings.warn(warn_str)
    if args.vis_attr_opt != "" and len(attr_methods) > 1:
        raise TypeError("The current implementation only supports a custom \"vis-attr-opt\" when only 1 attribution "
                        "method is used.")
    return attr_methods


def perturbate(x_input, nr_perturbations, pert_range):
    """ Given an input instance x_input, generate nr_perturbations perturbed instances around x_input. The distance from
    the input instance to any perturbed instance is pert_range at most."""
    # random values from -1 to 1
    rand_around_zero = ((torch.rand((nr_perturbations, x_input.shape[0])) * 2) - 1).to(x_input.device)
    # add random noise to the original instance, scaled to the maximum perturbation range
    x_noise = x_input.repeat((nr_perturbations, 1)) + (rand_around_zero * pert_range)

    # there are advantages and disadvantages with restricting the perturbed instances to certain bounds (for example
    # that no values can be smaller than 0), I decided to not restrict them
    # x_noise = torch.max(torch.zeros_like(x_noise), x_noise)

    # return both perturbations and the original input (the original input last)
    input_and_pert = torch.cat((x_noise, x_input.reshape(1, x_input.shape[0])))
    return input_and_pert


def apply_function(x_input, model, sigmoid=True):
    """ Return the model output. This function is not in line with the other attribution functions, which return a value
    for each feature. Here, only the output is returned. For this reason, it is not included in ALL_ATTR_METHODS. Still,
    it can be useful to show the model output in some scenarios (e.g. the "2D_special"/"2Dx" visualization)."""
    warn_str = "This method, indicated by the \"func\" identifier for attribution methods, is not an actual " \
               "attribution method but instead only returns the function output. In order to make it easier to use " \
               "the rest of the code, this is implemented in a way where the function output is returned as " \
               "\"attribution\" for the first feature, all other features remain 0. Note, that this is working as " \
               "intended, but it has to be used differently than the actual attribution methods. Visualization " \
               "methods will treat this as a normal attribution function which is fine as long as the user is aware " \
               "of what actually happens."
    warnings.warn(warn_str)
    output = torch.zeros_like(x_input)
    output[0, 0] = model(x_input, predict=sigmoid)
    return output


def attr_ig(x_input, model, bls, mpi=True, sigmoid=True):
    """ Apply integrated gradients attribution. If multiple baselines (bls) are given, every baseline is applied and the
    results are averaged. Options: baselines (bls), whether to multiply by inputs, so the global/local attribution (mpi)
    and whether to use the results of the NN after the sigmoid function or before (sigmoid). """
    attr_model = model

    ig = capttr.IntegratedGradients(attr_model, multiply_by_inputs=mpi)
    add_for_args = sigmoid
    n_steps = 50
    delta = 100  # just any value over the threshold of the following while loop
    attr_to_return = None  # will be set within the following while loop
    if not mpi:
        # use a large, hopefully sufficient number of steps, because we can not use delta as an error indicator here
        # it is possible to write some code to make delta work again, but that does not seem worth the effort to me
        # because not multiplying by inputs is the less interesting option anyway
        n_steps = 1000

    # 0.1 feels like a decent compromise between accurate results and acceptable runtime in general
    # keep in mind that this depends on the respective task, for example if there was a task which only outputs values
    # between 0 and 0.1, this delta threshold would not be very useful, for a task which outputs values in the
    # thousands or larger, this delta might even be a bit too strict
    # in general, this seems like a good value for the tasks I am currently looking at but it should be made sure that
    # the number of steps is sufficient for other tasks
    while abs(delta) > 0.1:
        attrs = []
        deltas = []
        # use every baseline (baselines have to be given in a list)
        for baseline in bls:
            # apply IG using captum
            attr, delta = ig.attribute(x_input, baselines=baseline, additional_forward_args=add_for_args,
                                       n_steps=n_steps, return_convergence_delta=True)
            attrs.append(attr)
            deltas.append(abs(delta.item()))
        # average attributions over baselines
        attr_to_return = torch.mean(torch.stack(attrs), dim=0)
        # calculate average delta
        delta = sum(deltas)/len(bls)
        if not mpi:
            delta = 0  # see comment above the while loop, the other if-condition for mpi
        n_steps *= 2
        if n_steps > 10000:
            print(f"IG delta: {delta}, keeping that result and not increasing number of steps further")
            break
    return attr_to_return


def attr_gshap(x_input, model, bls, mpi=True, sigmoid=True):
    """ Apply gradient SHAP attribution. Options: baselines (bls), whether to multiply by inputs, so the global/local
    attribution (mpi) and whether to use the results of the NN after the sigmoid function or before (sigmoid). """
    attr_model = model

    gradient_shap = capttr.GradientShap(attr_model, multiply_by_inputs=mpi)
    add_for_args = sigmoid

    # the same kind of loop increasing the number of samples depending on delta as for IG (there: number of steps) does
    # not work very well here (delta remains too large too often, probably because of the randomness in the calculation)
    # instead, 1000 samples are used every time

    # apply gradient SHAP using captum
    attr, delta = gradient_shap.attribute(x_input, baselines=bls, additional_forward_args=add_for_args,
                                          n_samples=1000, return_convergence_delta=True)

    return attr


def attr_sal(x_input, model, absolute=True, sigmoid=True):
    """ Apply saliency attribution. Options: whether to multiply by inputs, so the global/local attribution (mpi) and
    whether to use the results of the NN after the sigmoid function or before (sigmoid). """
    attr_model = model
    saliency = capttr.Saliency(attr_model)
    add_for_args = sigmoid
    # apply saliency using captum
    attr = saliency.attribute(x_input, abs=absolute, additional_forward_args=add_for_args)
    return attr


def attr_fp(x_input, model, sigmoid=True):
    """ Calculate attribution through feature permutation. Here, an instance generated near the input instance and then
    the input instance should be given in x_input (perturbation should already have happened). Then the attribution only
    for the input instance is calculated. Options: whether to use the results of the NN after the sigmoid function or
    before (sigmoid). """
    attr_model = model

    feature_perm = capttr.FeaturePermutation(attr_model)
    add_for_args = sigmoid
    # apply feature permutation using captum
    attr = feature_perm.attribute(x_input, additional_forward_args=add_for_args)
    # attribution for the original instance (not the perturbation)
    attr_to_return = attr[-1, :]
    attr_to_return = attr_to_return.reshape((1, attr_to_return.shape[0]))
    return attr_to_return


def attr_lime(x_input, model, pert_range=0.1, sigmoid=True):
    """ Apply attribution through LIME. There are many possible ways to apply LIME but I decided to use Ridge Regression
    with the functions as written here. Many other variants are simply less useful and give worse results but of course
    I can not rule out that other good ways to apply LIME exist. From those I came up with, what is written here had the
    best results. I only include one option for lime, namely the standard deviation when perturbing the data, as this
    has an important but still useful impact. Options: standard deviation when perturbing the data (std_div) and whether
    to use the results of the NN after the sigmoid function or before (sigmoid). """

    def similarity_kernel(
            original_input: torch.Tensor,
            perturbed_input: torch.Tensor,
            perturbed_interpretable_input: torch.Tensor,
            **kwargs) -> torch.Tensor:
        # calculate a similarity score for the original and the perturbed input (large similarity <-> large score)
        l2_dist = torch.norm(original_input - perturbed_input)
        # kernel width was part of the example in captum so I kept it
        # I think currently keeping it at 1 is sufficient but, depending on the task, playing around with this value
        # might also be interesting
        # TODO TODO TODO print("Specifc Kernel Width for large experiment (usually at 1)")
        # TODO ideally make the kernel width dependent on the data
        # kernel_width = 1
        kernel_width = 100
        # return l2_dist
        return torch.exp(- (l2_dist ** 2) / (kernel_width ** 2))

    def perturb_func(
            original_input: torch.Tensor,
            **kwargs) -> torch.Tensor:
        # generate data near the original input
        # get the input x
        o_inp_x = original_input
        # perturb x
        perturbed_x = perturbate(o_inp_x[0], 1, pert_range)[0:1]
        return perturbed_x

    def to_interp_rep_transform(
            curr_sample: torch.Tensor,
            original_input: torch.Tensor,
            **kwargs) -> torch.Tensor:
        # transform into interpretable space
        # especially considering low dimensional LPs, the dimensions are kept
        return curr_sample

    attr_model = model

    lime_attr = capttr.LimeBase(attr_model,
                                SkLearnLinearModel("linear_model.Ridge", alpha=0.1),
                                similarity_func=similarity_kernel,
                                perturb_func=perturb_func,
                                perturb_interpretable_space=False,
                                from_interp_rep_transform=None,
                                to_interp_rep_transform=to_interp_rep_transform)

    add_for_args = [sigmoid]
    # apply lime using captum
    attr = lime_attr.attribute(x_input, additional_forward_args=add_for_args, n_samples=50)
    return attr
