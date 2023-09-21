import deepxde as dde
import os
import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns


def divide_data(data):
    """
    Divide data into bc & collocation pts
    :param data: dde.data object to be used for obtaining collocation points
    :return: bc points & collocation points
    """
    num_bc_pts = data.train_x_bc.shape[0]
    bc_pts = data.train_x[:num_bc_pts]
    collocation_pts = data.train_x[num_bc_pts:]
    return bc_pts, collocation_pts

def update_collocation(model, data, N_adapt=0, type_adapt=0):
    """
    Update only collocation points of the given data (bc points do not change)
    :param data: dde.data object to be used for adaptive sampling
    :param N_adapt: collocation points to be newly added to the previous bc points
    :param criterion: criterion that determines which type of adaptive sampling will be used
            0: Vanilla adpative sampling (random sampling)
            1: Vorticity-Aware sampling
            2: advP-Aware sampling
            3: Both Vorticity-advP-Aware sampling
    :return: updated data
    """

    def get_new_collocation(model, eval_pts, N_adapt=100, type_adapt=0):
        """
        Extract top num_extraction points w.r.t. vorticity magnitude among collocation points
        :param model: trained model used for evaluating vorticity
        :param collocation_pts: location of the points used for evaluating vorticity
        :param N_adapt: number of points to be extracted w.r.t. vorticity magnitude
        :return: new set of collocation points (newly sampled collcoation pts + high vorticity magnitude pts)
        """
        def evaluate_vorticity(x, u):
            du_dy = dde.grad.jacobian(u, x, i=0, j=1)
            dv_dx = dde.grad.jacobian(u, x, i=1, j=0)
            return torch.abs(du_dy - dv_dx)

        def evaluate_advP(x, u):
            dP_dx = dde.grad.jacobian(u, x, i=2, j=0)
            return dP_dx

        if type_adapt==0 and N_adapt!=0:
            # In this case, size of the new collocation pts would decrease. Therefore, force N_adapt as 0 value.
            print("Though type_adapt is selected as random sampling, but N_adapt!=0 -> N_adapt is manully adjusted to 0!!")
            N_adapt = 0

        new_even = dde.data.PDE(data.geom, pde=None, bcs=[], num_domain=eval_pts.shape[0] - N_adapt,
                                train_distribution='LHS')
        new_even = new_even.train_x_all

        if type_adapt==0:
            """
            Vanilla adpative sampling (random sampling)
            Return only evenly sampled pts
            """
            return new_even

        elif type_adapt in [1, 2, 3]:

            if type_adapt==1:
                """
                Vorticity-Aware
                """
                y_evaluated = model.predict(eval_pts, operator=evaluate_vorticity).reshape(-1)
                sorted_indices = np.argsort(y_evaluated)[::-1]

                new_adapt = eval_pts[sorted_indices][:N_adapt]

            elif type_adapt==2:
                """
                advP-Aware
                """
                y_evaluated = model.predict(eval_pts, operator=evaluate_advP).reshape(-1)
                sorted_indices = np.argsort(y_evaluated)[::-1]

                new_adapt = eval_pts[sorted_indices][:N_adapt]

            elif type_adapt==3:
                """
                Both Vorticity-advP-Aware
                For this type, specialized algorithm is adopted.
                Step 1: Sort and extract N_adapt indicies w.r.t both V and P
                Step 2: extract overlapping and non-overlapping indices
                Step 3: new_adapt is defined by concatenating (overlapping pts, unique-v pts, unique-p pts)
                """
                # Step 1
                y_evaluated_v = model.predict(eval_pts, operator=evaluate_vorticity).reshape(-1)
                sorted_indices_v = np.argsort(y_evaluated_v)[::-1][:N_adapt]
                y_evaluated_p = model.predict(eval_pts, operator=evaluate_advP).reshape(-1)
                sorted_indices_p = np.argsort(y_evaluated_p)[::-1][:N_adapt]

                # Step 2
                overlap_indices, overlap_indices_v, overlap_indices_p = np.intersect1d(sorted_indices_v, sorted_indices_p, assume_unique=True, return_indices=True)
                # Get y_evaluated_v without overlapping with y_evaluated_p
                mask_v = np.ones(sorted_indices_v.size, dtype=bool)
                mask_v[overlap_indices_v] = False
                unique_v = sorted_indices_v[mask_v]
                # Get y_evaluated_p without overlapping with y_evaluated_v
                mask_p = np.ones(sorted_indices_p.size, dtype=bool)
                mask_p[overlap_indices_p] = False
                unique_p = sorted_indices_p[mask_p]

                # Step 3
                y_overlapped = eval_pts[overlap_indices]
                N_overlap = overlap_indices.size
                y_unique_v = eval_pts[unique_v][:int((N_adapt - N_overlap) / 2)]
                y_unique_p = eval_pts[unique_p][:N_adapt - N_overlap - int((N_adapt - N_overlap) / 2)]
                new_adapt = np.concatenate(
                    (y_overlapped,
                     y_unique_v,
                     y_unique_p)
                )

            return np.vstack((new_even, new_adapt))

        else:
            raise Exception("Invalid type for adaptive sampling!!")

    past_bc_pts, past_collocation_pts = divide_data(data)
    new_bc_pts = past_bc_pts # Same bc pts will be used (bc pts are not updated)

    if True:
        # 새롭게 샘플링된 pts를 사용하여 vorticity evaluation
        eval_pts = dde.data.PDE(data.geom, pde=None, bcs=[],
                                num_domain=past_collocation_pts.shape[0],
                                train_distribution='LHS').train_x_all
    else:
        # 이전 iteration의 collocation pts를 사용하여 vorticity evaluation
        eval_pts = past_collocation_pts

    new_collocation_pts = get_new_collocation(model, eval_pts, N_adapt=N_adapt, type_adapt=type_adapt)

    new_train_x = np.vstack((new_bc_pts, new_collocation_pts))

    data.train_x = new_train_x
    print(f"Adaptive sampling for type {type_adapt} is completed!")

def plot_pts(data, N_adapt=100, cur_directory="", tag="0"):

    past_x = data.train_x[:-N_adapt][:,0]
    past_y = data.train_x[:-N_adapt][:,1]

    added_x = data.train_x[-N_adapt:][:,0]
    added_y = data.train_x[-N_adapt:][:,1]

    fig, ax = plt.subplots(dpi=150)
    ax.scatter(past_x, past_y, alpha=0.5, color='k', label="Even collocation points")
    ax.scatter(added_x, added_y, alpha=0.7, color='r', label="Vorticity-aware collocation points")
    ax.legend(fontsize=15, loc='lower right', frameon=True)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    fig_index = 0
    while True:
        fig_index += 1
        cur_dir = os.path.join(os.getcwd(), f"fig\Pts_{tag}_{fig_index}.png")
        if os.path.isfile(cur_dir):
            continue
        else:
            fig.savefig(f"./fig/Pts_{tag}_{fig_index}")
            break
    # plt.show()

def plot_flowfield(x1, x2, y1, y2, qoi_name=["U_x","U_y"], tag='Vanilla', stream=True):
    y1 = y1.reshape(len(x2), len(x1))
    y2 = y2.reshape(len(x2), len(x1))
    fig, ax = plt.subplots(dpi=150)
    if stream:
        ax.streamplot(x1, x2, y1, y2, density=1.5, linewidth=0.7, color='w', arrowsize=0.7, broken_streamlines=True)
    img = ax.contourf(x1, x2, y1, levels = np.linspace(-0.2, 1., 100), cmap=sns.color_palette("icefire", as_cmap=True), extend='both')
    fig.colorbar(img, ticks=np.linspace(img.levels.min(),img.levels.max(),6))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title(f"{tag}:${qoi_name[0]} [m/s]$", fontsize=13)
    if stream:
        fig.savefig(f"./fig/{qoi_name[0]}_{tag}_stream")
    else:
        fig.savefig(f"./fig/{qoi_name[0]}_{tag}")
    # plt.show()

    fig, ax = plt.subplots(dpi=150)
    if stream:
        ax.streamplot(x1, x2, y1, y2, density=1.5, linewidth=0.7, color='w', arrowsize=0.7, broken_streamlines=True)
    img = ax.contourf(x1, x2, y2, levels = np.linspace(-0.5, .3, 100), cmap=sns.color_palette("icefire", as_cmap=True), extend='both')
    fig.colorbar(img, ticks=np.linspace(img.levels.min(),img.levels.max(),6))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title(f"{tag}:${qoi_name[1]} [m/s]$", fontsize=13)
    if stream:
        fig.savefig(f"./fig/{qoi_name[1]}_{tag}_stream")
    else:
        fig.savefig(f"./fig/{qoi_name[1]}_{tag}")
    # plt.show()

def eval_pde_loss(model, x_eval=None):
    if x_eval is None:
        x_, y_ = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
        x_eval = np.concatenate((x_.reshape(-1, 1), y_.reshape(-1, 1)), axis=1)
    if x_eval.dtype is not torch.float32:
        x_eval = torch.tensor(x_eval, dtype=torch.float32)
    y_target = model.predict(x_eval.to("cpu"))
    # first element of the output_losses_test function is predicted values (that is, u, v, p)
    # second element is the mean losses (axis=1)
    losses = model.outputs_losses_test(x_eval, y_target, auxiliary_vars=None)[1].detach()
    return losses