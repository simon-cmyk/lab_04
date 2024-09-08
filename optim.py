import numpy as np


def gauss_newton(x_init, model, cost_thresh=1e-14, delta_thresh=1e-14, max_num_it=15):
    """Implements nonlinear least squares using the Gauss-Newton algorithm

    :param x_init: The initial state
    :param model: Model with a function linearise() the returns A, b and the cost for the current state estimate.
    :param cost_thresh: Threshold for cost function
    :param delta_thresh: Threshold for update vector
    :param max_num_it: Maximum number of iterations
    :return:
      - x: State estimates at each iteration, the final state in x[-1]
      - cost: The cost at each iteration
      - A: The full measurement Jacobian at the final state
      - b: The full measurement error at the final state
    """
    x = [None] * (max_num_it + 1)
    cost = np.zeros(max_num_it + 1)

    x[0] = x_init
    for it in range(max_num_it):
        A, b, cost[it] = model.linearise(x[it])
        A_ = A.T @ A
        b_ = A.T @ b
        tau = np.linalg.lstsq(A, b, rcond=None)[0]
        x[it + 1] = x[it] + tau

        if cost[it] < cost_thresh or np.linalg.norm(tau) < delta_thresh:
            x = x[:it + 2]
            cost = cost[:it + 2]
            break

    A, b, cost[-1] = model.linearise(x[-1])

    return x, cost, A, b

def Levenberg_Marquart(x_init, model, cost_thresh=1e-14, delta_thresh=1e-14, max_num_it=10):
    """Implements nonlinear least squares using the LM algorithm
    """
    lam = 1e-4 

    x = [None] * (max_num_it + 1)
    cost = np.zeros(max_num_it + 1)

    x[0] = x_init
    for it in range(max_num_it):
        A, b, cost[it] = model.linearise(x[it])
        AtA = A.T @ A

        A_star = AtA + lam * np.eye(AtA.shape[0])
        b_star = A.T @ b

        tau = np.linalg.lstsq(A_star, b_star, rcond=None)[0]
        x[it + 1] = x[it] + tau

        _, _, cost_ = model.linearise(x[it + 1])
        if  cost_ < cost[it]:
            lam /= 10
        else:
            lam *= 10
            x[it + 1] = x[it]

        if cost[it] < cost_thresh or np.linalg.norm(tau) < delta_thresh:
            x = x[:it + 2]
            cost = cost[:it + 2]
            break

    A, b, cost[-1] = model.linearise(x[-1])

    return x, cost, A, b

