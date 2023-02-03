""" I used an mnist examples as a starting point to not have to implement everything from scratch and changed that for
the purpose of training a neural network for a single LP.
(reference: https://github.com/pytorch/examples/blob/master/mnist/main.py, accessed 16.07.2021) """
import sys
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader

import util
import attributions as attribs
import visualize as vis
from lp_encodings import change_feasible_encoding

from rtpt import RTPT


class SingleLPNet(nn.Module):
    """ Neural network representing a single LP. Here, a number of instances in the problem space are given as input and
    the corresponding binary output describes whether the respective instance is within the feasible region of the LP
    or not. This neural network does not capture any information about the optimal solution (at least not directly). """

    def __init__(self, input_dimension):
        """ Initialize the neural network. """
        super(SingleLPNet, self).__init__()
        # after trying out multiple configurations, the following number of neurons and layers gave the best results out
        # of those configurations tried out
        # while those still had the best results on the train set, it might be better to have a simpler, smaller
        # architecture instead (for the 2-dimensional problem in the paper, the architecture does not have to be so
        # large but still, this is the one which is used)
        hidden_neurons = 2**12
        self.fc1 = nn.Linear(input_dimension, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc4 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc5 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc6 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc7 = nn.Linear(hidden_neurons, 1)

    def forward(self, x, predict=False):
        """ Forward pass. """
        # print(predict)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.fc2(x)
        x = func.relu(x)
        x = self.fc3(x)
        x = func.relu(x)
        x = self.fc4(x)
        x = func.relu(x)
        x = self.fc5(x)
        x = func.relu(x)
        x = self.fc6(x)
        x = func.relu(x)
        x = self.fc7(x)
        # keep in mind, that predict should never be True if the encoding is not "Feasibility"
        if not predict:
            return x
        else:
            return torch.sigmoid(x)


class SingleLPNetCustom(SingleLPNet):
    # TODO description and comments, also I think this subclass style is bad, but as long as it works as intended...

    def __init__(self, input_dimension, arch_info):
        """ Initialize the neural network. """
        super(SingleLPNet, self).__init__()  # do not use super(SingleLPNetCustom) to not create unnecessary layers
        self.arch_info = arch_info
        nr_layers, nr_neurons, neuron_dev = self.arch_info.split("-")
        nr_layers = int(nr_layers)
        nr_neurons = int(nr_neurons)
        self.layers = nn.ModuleList()
        prev_neurons = input_dimension
        next_neurons = nr_neurons
        for layer in range(nr_layers):
            self.layers.append(nn.Linear(prev_neurons, next_neurons))
            if next_neurons != 1:
                self.layers.append(nn.Dropout(p=0.5))
            prev_neurons = next_neurons
            if layer == (nr_layers - 2):
                next_neurons = 1
            elif neuron_dev == "dec":
                next_neurons = int(0.5*next_neurons)
            # TODO more neuron_devs?
            # next_neurons can stay unchanged for neuron_dev == "cons"

            # if layer == 0:
            #     self.layers.append(nn.Linear(input_dimension, nr_neurons))
            # elif layer == (nr_layers - 1):
            #     self.layers.append(nn.Linear(nr_neurons, 1))
            # else:
            #     self.layers.append(nn.Linear(nr_neurons, nr_neurons))

    def forward(self, x, predict=False):
        for l_id, layer in enumerate(self.layers):
            x = layer(x)
            if l_id == len(self.layers)-1:
                # we do not want a relu on the last layer
                break
            if type(self.layers[l_id+1]) == torch.nn.modules.linear.Linear:
                x = func.relu(x)
        # keep in mind, that predict should never be True if the encoding is not "Feasibility"
        if not predict:
            return x
        else:
            return torch.sigmoid(x)


def train(args, model, device, train_loader, optimizer, epoch):
    """ Train the model. """
    model.train()
    if args.enc == "Feasibility":
        # binary cross entropy loss with sigmoid
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        # mse loss
        criterion = torch.nn.MSELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            if args.dry_run:
                break


def test(args, model, device, test_loader):
    """ Test the model. """
    model.eval()
    test_loss = 0
    correct = 0
    if args.enc == "Feasibility":
        # binary cross entropy loss with sigmoid
        criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    else:
        # mse loss
        criterion = torch.nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = torch.round(torch.sigmoid(output))
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.batch_sampler)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(
                                                                                     test_loader.dataset)))
    return test_loss


def train_model(args):
    """ Get model parameters, data and train a model. """
    rtpt = RTPT(name_initials="FB", experiment_name="xlp_preparation", max_iterations=args.epochs)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1,
                       "pin_memory": True,
                       "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    rtpt.start()
    c, a, b, x, y, sol = util.get_data_single_lp(args)
    if args.enc != "Feasibility":
        y = change_feasible_encoding(args.enc, c, a, b, x, y)
    model_input_dim = x.shape[1]

    # use half of the data for training and testing each
    x_train = x[:x.shape[0] // 2]
    x_test = x[x.shape[0] // 2:]

    y_train = y[:x.shape[0] // 2]
    y_test = y[x.shape[0] // 2:]

    tensor_x_train = torch.tensor(x_train, dtype=torch.float32)    # TODO Tensor previously without dtype
    tensor_x_test = torch.tensor(x_test, dtype=torch.float32)      # TODO Tensor previously without dtype
    tensor_y_train = torch.tensor(y_train, dtype=torch.float32)    # TODO Tensor previously without dtype
    tensor_y_test = torch.tensor(y_test, dtype=torch.float32)      # TODO Tensor previously without dtype

    dataset_train = TensorDataset(tensor_x_train, tensor_y_train)
    dataset_test = TensorDataset(tensor_x_test, tensor_y_test)

    dataloader_train = DataLoader(dataset_train, **train_kwargs)
    dataloader_train_for_test = DataLoader(dataset_train, **test_kwargs)
    dataloader_test = DataLoader(dataset_test, **test_kwargs)

    model = SingleLPNet(model_input_dim) if args.architecture == "" else SingleLPNetCustom(model_input_dim, args.architecture)
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    train_loss = []
    test_loss = []
    rtpt.step()
    rtpt = RTPT(name_initials="FB", experiment_name="xlp_training", max_iterations=args.epochs)
    rtpt.start()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, dataloader_train, optimizer, epoch)
        train_loss.append(test(args, model, device, dataloader_train_for_test))
        test_loss.append(test(args, model, device, dataloader_test))
        scheduler.step()
        rtpt.step()  # (subtitle=f"{epoch}/{args.epochs}")
    vis.visualize_loss(train_loss, test_loss, args.save_name)

    if args.save_model:
        path = "models/"
        # save the model
        save_path = f"{path}{args.save_name}.pt"
        torch.save(model.state_dict(), save_path)
        if args.no_data_storing:
            info = util.make_info(args, c=c, a=a, b=b, x=x, y=y, sol=sol)
        else:
            # save other info
            info = util.store_info(path, args, c=c, a=a, b=b, x=x, y=y, sol=sol)
    else:
        info = util.make_info(args, c=c, a=a, b=b, x=x, y=y, sol=sol)

    return model, info


def prepare_model(args):
    """ Define the model and load the state in the specified path. """
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load one template datapoint just to get the model dimension
    c, a, b, x, y, sol = util.get_data_single_lp(args)  # TODO add again (together with below), num_x=0)
    model_input_dim = x.shape[1]
    model = SingleLPNet(model_input_dim).to(device) if args.architecture == "" else SingleLPNetCustom(model_input_dim, args.architecture).to(device)
    model = nn.DataParallel(model)
    model.to(device)

    # load the model state
    path = "models/"
    save_path = f"{path}{args.save_name}.pt"
    model.load_state_dict(torch.load(save_path))
    if args.no_data_storing:
        # regenerate other data (random seed is used)
        # TODO add again (together with above) c, a, b, x, y, sol = util.get_data_single_lp(args)
        info = util.make_info(args, c=c, a=a, b=b, x=x, y=y, sol=sol)
    else:
        # load other info
        info = util.load_info(path, args.save_name)

    return model, info


def vis_special(model, x, a, b, attr_methods, vis_save, device, args, full_x_and_dims=None, fix_base_IG=False, name_add=""):
    """ This "special" visualization task does not use any individual points for the attribution but instead considers
    the whole data space. This is only implemented to work for 2-dimensional data (2 features) which can then be
    visualized in a 2D image. The final visualization consists of three plots. The first plot shows the overall
    attribution sum, the second and third plot show the attribution for the first and second feature, respectively. """
    x_max = x.max(axis=0)[0]
    # do not be confused: here, y indicates the second axis in the 2d visualization, not the (non-)feasible label
    y_max = x.max(axis=0)[1]

    # nr_steps determines the precision of the background color (how many points are calculated)
    # this number of steps will be the maximum precision for each dimension
    # for the smaller dimension, the number of steps might be smaller, so that each resulting point is about as high as
    # it is wide
    nr_steps = args.vis_special_steps
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

    # x_needed contains the points for which an attribution will be calculated
    x_needed = torch.zeros((x_steps * y_steps, 2))
    for i in range(x_steps):
        for j in range(y_steps):
            x_needed[i * y_steps + j] = torch.tensor([(i + 0.5) * one_x_step, (j + 0.5) * one_y_step])    # TODO Tensor previously
    x_needed = x_needed.to(device)
    # full_x_and_dims is always None, except for large visualizations (where only parts of the LP are visualized)
    if full_x_and_dims is not None:
        full_x = full_x_and_dims[0]
        full_dims = full_x_and_dims[1]
        # for large visualizations, the other features are kept constant as given by full_x and only the two features
        # chosen for the visualization (given by full_dims) are varied and plotted
        x_temp = torch.Tensor(full_x).to(device).repeat(x_needed.shape[0], 1)
        x_temp[:, full_dims] = x_needed
        x_needed = x_temp

    # calculate attribution for x_needed
    for attr_method in attr_methods:
        options = attribs.get_options(attr_method, args.vis_attr_opt)
        if args.enc != "Feasibility" and len(options) != 0:
            if options[-1] is True:
                print("There only exists one possibility for the last attribution option (no sigmoid)")
                options = (*options[:-1], False)
        x_attr = torch.zeros_like(x_needed)
        for i in range(x_needed.shape[0]):
            if attr_method == "func":
                # model output (no attribution)
                x_attr[i] = attribs.apply_function(x_needed[i: i + 1], model, sigmoid=options[0])
            if attr_method == "ig":
                # integrated gradients
                if full_x_and_dims is not None and fix_base_IG:
                    # this option is only for large visualizations
                    # here, the baseline for IG is being set to equal the input point ("fixed") for all but the two
                    # features considered for the visualizations (otherwise, all features are set according to the
                    # baseline option; if there are only two features, these approaches do not differ)
                    default_bl = x_needed[i: i + 1].detach().clone()
                    if options[1] == 0:
                        default_bl[:, full_dims] = 0
                        bls = [default_bl]
                    elif options[1] == 1:
                        default_bl[:, full_dims] = 0
                        bls = [default_bl]
                        default_bl2 = x_needed[i: i + 1].detach().clone()
                        default_bl2[0, full_dims] = torch.tensor(x.max(axis=0)[full_dims],
                                                                 dtype=torch.float32).to(device)
                        bls += [default_bl2]
                    else:
                        raise ValueError(f"There is not baseline option {options[1]}.")
                else:
                    if options[1] == 0:
                        bls = [torch.zeros_like(x_needed[i:i + 1])]
                    elif options[1] == 1:
                        bls = [torch.zeros_like(x_needed[i:i + 1])]
                        bls += [torch.tensor(x.max(axis=0).reshape((1, x.shape[1]))).to(device)]   # TODO Tensor previously
                    else:
                        raise ValueError(f"There is not baseline option {options[1]}.")
                x_attr[i] = attribs.attr_ig(x_needed[i: i + 1], model, bls, mpi=options[0], sigmoid=options[2])
            if attr_method == "gshap":
                # gradient SHAP
                if options[1] == 0:
                    bls = torch.zeros_like(x_needed[i:i + 1]).to(device)
                elif options[1] == 1:
                    edge_low_x = torch.zeros_like(x_needed[i:i + 1]).to(device)
                    edge_high_x = torch.tensor(x.max(axis=0)).unsqueeze(0).to(device)
                    bls = torch.cat((edge_low_x, edge_high_x)).float()
                elif options[1] == 2:
                    bls = (torch.rand(100, x.shape[1]) * torch.tensor(x.max(axis=0), dtype=torch.float32)).to(device)
                else:
                    raise ValueError(f"There is not baseline option {options[1]}.")
                x_attr[i] = attribs.attr_gshap(x_needed[i: i + 1], model, bls, mpi=options[0], sigmoid=options[2])
            if attr_method == "sal":
                # saliency
                x_attr[i] = attribs.attr_sal(x_needed[i: i + 1], model, absolute=options[0], sigmoid=options[1])
            if attr_method == "fp":
                # feature permutation method (generate one other instance and use attribution with respect to input
                # instance, but do this options[0] times)
                attr_temp = torch.zeros_like(x_attr[i:i+1])
                for j in range(options[0]):
                    # create 1 data point around the input and apply permutation
                    input_for_attr = attribs.perturbate(x_needed[i], 1, options[1])
                    attr_temp += attribs.attr_fp(input_for_attr, model, sigmoid=options[2])
                x_attr[i] = attr_temp/options[0]
            if attr_method == "lime":
                # lime
                x_attr[i] = attribs.attr_lime(x_needed[i: i + 1], model, pert_range=options[0], sigmoid=options[1])
            if (i+1) % 100 == 0:
                print(f"Finished {i+1} of {x_needed.shape[0]} calculations")
        enc_str = args.enc + "_"
        name = f"{enc_str}{attr_method}_{str(args.vis_attr_opt)}" if vis_save else ""
        if name_add != "":
            name += "_" + name_add
        attr = x_attr.detach().cpu().numpy()
        if full_x_and_dims is not None:
            # remove everything but the two relevant features for the visualization
            attr = attr[:, full_dims]
            x_needed = x_needed[:, full_dims]
        vis.visualize_single_lps("2Dx", attr, a=a, b=b, x=x_needed.detach().cpu().numpy(), name=name)


def apply_visualization(model, info, args):
    """ Set all necessary parameters and call the right visualization method. """
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    rtpt = RTPT(name_initials="FB", experiment_name="xlp_evaluation", max_iterations=args.num_vis)
    rtpt.start()

    # allow user input for the input (instead of using test data)
    custom_input = args.vis_input

    # apply special visualization (see 'vis_special' function for more details)
    apply_special_vis = args.vis_special

    # apply visualization for large lps (only parts of the LP and attribution are visualized)
    apply_large_vis = args.vis_large
    if apply_large_vis:
        # extract several options for the  large visualization
        small_x_dim, small_b_dim, largest_vio, fix_base_IG = args.vis_large_options.split("-")
        # how many dimensions of x to visualize
        small_x_dim = int(small_x_dim)
        # how many constraints (dimensions of b) to visualize
        small_b_dim = int(small_b_dim)
        # whether to use the largest constraint violation to select the features to be visualized
        # if False, the violated constraints closest to not being violated are selected
        largest_vio = True if largest_vio == "1" else False
        # whether to fix the IG baseline for the features not be visualized in the 2d visualization
        fix_base_IG = True if fix_base_IG == "1" else False

    # get data to look at for the attribution and visualization
    c, a, b, x, y, sol = info.get_data()
    x_train = x[x.shape[0] // 2:]
    x_test = x[x.shape[0] // 2:]
    start_index = args.num_vis * args.vis_next
    if start_index + args.num_vis > x_test.shape[0]:
        raise ValueError(f"There are not enough test instances to visualize with respect to \"args.num_vis\": "
                         f"{args.num_vis} and \"args.vis_next\": {args.vis_next}")
    x_test = x_test[start_index:start_index + args.num_vis]
    tensor_x_vis = torch.tensor(x_test, dtype=torch.float32).to(device)    # TODO Tensor previously

    # which attribution methods to apply
    attr_methods = attribs.parse_methods(args)

    if apply_special_vis:
        vis_special(model, x, a, b, attr_methods, args.vis_save, device, args)
    else:
        for i in range(args.num_vis):
            if custom_input:
                input_str = input(f"Enter {x.shape[1]} whitespace separated int or float values.")
                input_strs = input_str.split()
                if len(input_strs) != x.shape[1]:
                    raise ValueError(f"Need to enter {x.shape[1]} values.")
                for count, inp_str in enumerate(input_strs):
                    tensor_x_vis[i: i + 1, count] = float(inp_str)
                x_vis = tensor_x_vis[i: i + 1].detach().cpu().numpy()
            else:
                x_vis = x_test[i:i + 1]

            def check_options(opts):
                # resolve a possible conflict with a differently encoded SingleLP
                if args.enc != "Feasibility" and len(opts) != 0:
                    if opts[-1] is True:
                        print("There only exists one possibility for the last attribution option (no sigmoid)")
                        return (*opts[:-1], False)
                return opts

            torch.manual_seed(args.seed)

            # get the model predictions
            if args.enc == "Feasibility":
                pred = model(tensor_x_vis[i: i + 1], predict=True)
            else:
                pred = model(tensor_x_vis[i: i + 1])
            pred = pred.detach().cpu().numpy()

            def finish_vis():
                if not apply_large_vis:
                    vis.visualize_single_lps(args.problem, attr, c, a, b, x_vis, pred, name=name, x_opt=sol)
                else:
                    two_inds = vis.visualize_large_lp(attr, a, b, x_vis, pred, name, small_x_dim, small_b_dim, True,
                                                      largest_vio)
                    two_inds.sort()  # might be unnecessary, but can not hurt either
                    x_vis[0, two_inds] = 0
                    # update the constraint with all but the two features (two_inds) given as constant
                    # use tensor just because I do not want to import numpy for one line
                    b_2d = b - torch.matmul(torch.Tensor(a), torch.Tensor(x_vis.T)).numpy()[:, 0]
                    # visualize the two selected features
                    # deactivatet the following part as it was not useful most of the time but takes a lot of time
                    if False:  # "fp" not in attr_methods:  # fp takes too much time for large lps
                        name_add = "f_" + "_".join(str(two_inds_elem) for two_inds_elem in two_inds) + "_nr_" + str(i)
                        vis_special(model, x[:, two_inds], a[:, two_inds], b_2d, attr_methods, args.vis_save, device, args,
                                    full_x_and_dims=(x_vis, two_inds), fix_base_IG=fix_base_IG, name_add=name_add)

            if "ig" in attr_methods:
                # integrated gradients
                options = attribs.get_options("ig", args.vis_attr_opt)
                options = check_options(options)
                if options[1] == 0:
                    bls = [torch.zeros_like(tensor_x_vis[i:i + 1])]
                elif options[1] == 1:
                    bls = [torch.zeros_like(tensor_x_vis[i:i + 1])]
                    bls += [torch.tensor(x_train.max(axis=0).reshape((1, x.shape[1])), dtype=torch.float32).to(device)]  # TODO Tensor previously
                else:
                    raise ValueError(f"There is not baseline option {options[1]}.")
                attr = attribs.attr_ig(tensor_x_vis[i: i + 1], model, bls, mpi=options[0], sigmoid=options[2])
                attr = attr.detach().cpu().numpy()
                name = f"{args.problem}_{i}_IG" if args.vis_save else ""
                finish_vis()

            if "gshap" in attr_methods:
                # gradient SHAP
                options = attribs.get_options("gshap", args.vis_attr_opt)
                options = check_options(options)
                if options[1] == 0:
                    bls = torch.zeros((1, x.shape[1])).to(device)
                elif options[1] == 1:
                    edge_low_x = torch.zeros((1, x.shape[1])).to(device)
                    edge_high_x = torch.tensor(x_train.max(axis=0)).unsqueeze(0).to(device)
                    bls = torch.cat((edge_low_x, edge_high_x)).float()
                elif options[1] == 2:
                    x_max = torch.tensor(x_train.max(axis=0), dtype=torch.float32)
                    bls = (torch.rand(100, x.shape[1]) * x_max).to(device)
                else:
                    raise ValueError(f"There is not baseline option {options[1]}.")
                attr = attribs.attr_gshap(tensor_x_vis[i: i + 1], model, bls, mpi=options[0], sigmoid=options[2])
                attr = attr.detach().cpu().numpy()
                name = f"{args.problem}_{i}_GS" if args.vis_save else ""
                finish_vis()

            if "sal" in attr_methods:
                # saliency
                options = attribs.get_options("sal", args.vis_attr_opt)
                options = check_options(options)
                attr = attribs.attr_sal(tensor_x_vis[i: i + 1], model, absolute=options[0], sigmoid=options[1])
                attr = attr.detach().cpu().numpy()
                name = f"{args.problem}_{i}_SAL" if args.vis_save else ""
                finish_vis()

            if "fp" in attr_methods:
                # feature permutation
                options = attribs.get_options("fp", args.vis_attr_opt)
                options = check_options(options)
                attr_temp = torch.zeros_like(tensor_x_vis[i: i + 1])
                for j in range(options[0]):
                    # create 1 data point around the input and apply permutation
                    input_for_attr = attribs.perturbate(tensor_x_vis[i], 1, options[1])
                    attr_temp += attribs.attr_fp(input_for_attr, model, sigmoid=options[2])
                attr = attr_temp/options[0]
                attr = attr.detach().cpu().numpy()
                name = f"{args.problem}_{i}_FP" if args.vis_save else ""
                finish_vis()

            if "lime" in attr_methods:
                # lime
                options = attribs.get_options("lime", args.vis_attr_opt)
                options = check_options(options)
                attr = attribs.attr_lime(tensor_x_vis[i: i + 1], model, pert_range=options[0], sigmoid=options[1])
                attr = attr.detach().cpu().numpy()
                name = f"{args.problem}_{i}_LIME" if args.vis_save else ""
                finish_vis()
            
            rtpt.step()


def prepare_arguments():
    """ Define and return arguments. """
    parser = ArgumentParser(description="PyTorch Single LP Experiment")
    # model training
    parser.add_argument("--architecture", type=str, default="", metavar="ID",
                        help="use different NN architectures, see code for details on how input is parsed and used")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=5, metavar="N", help="number of epochs to train (default: 5)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed (default: 0)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    # model saving / loading
    parser.add_argument("--save-model", action="store_true", default=False, help="save the current model")
    parser.add_argument("--load-model", action="store_true", default=False, help="load a model")
    parser.add_argument("--save-name", type=str, default="SingleLPNet", metavar="NAME",
                        help="name with which the model will be saved or loaded")
    parser.add_argument("--no-data-storing", action="store_true", default=False,
                        help="only save the model, not the data created for training the model")
    # linear problem type
    parser.add_argument("--problem", type=str, default="SingleLP", metavar="ID", help="specify a specific LP")
    parser.add_argument("--dim-x", type=int, default=5, metavar="N", help="dimension of instance in LP")
    parser.add_argument("--dim-b", type=int, default=3, metavar="N", help="dimension of constrains b of the LP")
    parser.add_argument("--num-x", type=int, default=10**6, metavar="N", help="number of generated instances")
    parser.add_argument("--enc", type=str, default="Feasibility", metavar="ID", help="specify the encoding of the LP")
    # visualization
    parser.add_argument("--vis", action="store_true", default=False, help="visualize model performance and attribution")
    parser.add_argument("--num-vis", type=int, default=10, metavar="N", help="number of instanced to be visualized")
    parser.add_argument("--vis-next", type=int, default=0, metavar="N",
                        help="skips the first vis_next * num_vis instances, can visualize other instances that way")
    parser.add_argument("--vis-save", action="store_true", default=False,
                        help="save the visualization, otherwise simply show it")
    parser.add_argument("--vis-attr", type=str, default="all", metavar="ID", help="which attribution methods to apply, "
                        "multiple methods can be separated with \"-\", \"all\" indicates applying all implemented "
                        "methods")
    parser.add_argument("--vis-attr-opt", type=str, default="", metavar="ID",
                        help="identifier for different options for the attribution methods")
    parser.add_argument("--vis-input", action="store_true", default=False,
                        help="enter own inputs for the visualization")
    parser.add_argument("--vis-special", action="store_true", default=False, help="apply special 2d visualization")
    parser.add_argument("--vis-special-steps", type=int, default=100, metavar="N",
                        help="visualization precision/steps for vis-special visualization on larger dimension")
    parser.add_argument("--vis-large", action="store_true", default=False, help="apply visualization for large LPs")
    parser.add_argument("--vis-large-options", type=str, default="3-3-1-1", metavar="ID",
                        help="identifier for different options for the large LP visualizations")  # TODO explain what each means

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    return args


def main(overwrite_argv=None):
    """ Run the neural network and visualization with the specified arguments. """
    # overwrite the argument vector, which makes it so that the arguments for running the file are ignored
    # they are not overwritten when running this file (SingleLP.py) but can be overwritten when calling this file from
    # an external source, making it easier to set specific configurations in code
    if overwrite_argv is not None:
        new_argv = [sys.argv[0]] + overwrite_argv
        sys.argv = new_argv

    # get arguments
    args = prepare_arguments()

    # get the model
    if not args.load_model:
        # train the model
        model, info = train_model(args)
    else:
        # load the model
        model, info = prepare_model(args)

    # obtain and visualize attributions
    if args.vis:
        apply_visualization(model, info, args)


if __name__ == "__main__":
    main()
