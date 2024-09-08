import numpy as np
import visgeom as vg
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
from pylie import SO3, SE3
from optim import gauss_newton, Levenberg_Marquart

"""Example 2 - MAP estimation"""

np.random.seed(42)

class NoisyPointAlignmentBasedPoseEstimatorObjective:
    """Implements linearisation of the covariance weighted objective function"""

    def __init__(self, x_w, x_o, covs, T_prior=None, Cov_prior=None):
        if x_w.shape[0] != 3 or x_w.shape != x_o.shape:
            raise TypeError('Matrices with corresponding points must have same size')

        self.x_w = x_w
        self.x_o = x_o
        self.num_points = x_w.shape[1]

        if len(covs) != self.num_points:
            raise TypeError('Must have a covariance matrix for each point observation')

        # Compute square root of information matrices.
        self.sqrt_inv_covs = [None] * self.num_points
        for i in range(self.num_points):
            self.sqrt_inv_covs[i] = scipy.linalg.sqrtm(scipy.linalg.inv(covs[i]))

        self.T_prior = T_prior  # Mean of the prior
        self.Cov_prior = Cov_prior  # Covariance of the prior

        if T_prior is not None and Cov_prior is not None:
            self.sqrt_inv_cov_prior = scipy.linalg.sqrtm(scipy.linalg.inv(Cov_prior))
        else:
            self.sqrt_inv_cov_prior = None

    def linearise(self, T_wo):
        A = np.zeros((3 * self.num_points, 6))
        b = np.zeros((3 * self.num_points, 1))
        T_wo_inv = T_wo.inverse()

        # Enter the submatrices from each measurement:
        for i in range(self.num_points):
            A[3 * i:3 * (i + 1), :] = self.sqrt_inv_covs[i] @ \
                                      T_wo_inv.jac_action_Xx_wrt_X(self.x_w[:, [i]]) @ T_wo.jac_inverse_X_wrt_X()
            b[3 * i:3 * (i + 1)] = self.sqrt_inv_covs[i] @ (self.x_o[:, [i]] - T_wo_inv * self.x_w[:, [i]])

        if self.T_prior is not None and self.Cov_prior is not None:
            A_prior = self.sqrt_inv_cov_prior @ np.eye(6)  # Identity matrix for pose parameters
            b_prior = self.sqrt_inv_cov_prior @ (T_wo.Log() - self.T_prior.Log())  # Pose difference in tangent space
            
            # Stack prior terms to A and b
            A = np.vstack((A, A_prior))
            b = np.vstack((b, b_prior))


        return A, b, b.T.dot(b)


def main():
    # World box.
    points_w = vg.utils.generate_box()

    # True observer pose.
    true_pose_wo = SE3((SO3.rot_z(np.pi), np.array([[3, 3, 0]]).T))

    # Observed box with noise.
    points_o = vg.utils.generate_box(pose=true_pose_wo.inverse().to_tuple())
    num_points = points_o.shape[1]
    point_covariances = [np.diag(np.array([1e-1, 1e-1, 1e-1]) ** 2)] * num_points
    for c in range(num_points):
        points_o[:, [c]] = points_o[:, [c]] + np.random.multivariate_normal(np.zeros(3), point_covariances[c]).reshape(
            -1, 1)

    # Perturb observer pose and use as initial state.
    init_pose_wo = true_pose_wo + 1 * np.random.randn(6, 1)

    prior_mean = true_pose_wo
    prior_cov = np.diag(np.array([1, 1, 1, 1, 1, 1]) ** 2)

    # Estimate pose in the world frame from point correspondences.
    model = NoisyPointAlignmentBasedPoseEstimatorObjective(points_w, points_o, point_covariances)
    x, cost, A, b = Levenberg_Marquart(init_pose_wo, model, max_num_it=100)
    cov_x_final = np.linalg.inv(A.T @ A)

    # Print covariance.
    with np.printoptions(precision=3, suppress=True):
        print('Covariance:')
        print(cov_x_final)

    # Visualize (press a key to jump to the next iteration).
    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot box and true state
    vg.plot_pose(ax, true_pose_wo.to_tuple(), scale=1, alpha=0.4)
    vg.utils.plot_as_box(ax, points_w, alpha=0.4)

    # Plot initial state (to run axis equal first time).
    ax.set_title('Cost: ' + str(cost[0]))
    artists = vg.plot_pose(ax, x[0].to_tuple(), scale=1)
    artists.extend(vg.utils.plot_as_box(ax, x[0] * points_o))
    vg.plot.axis_equal(ax)
    plt.draw()
    print(f'final error (tangent space) {np.linalg.norm(true_pose_wo.Log()- x[-1].Log())}')
    while True:
        if plt.waitforbuttonpress():
            break

    # Plot iterations
    for i in range(1, len(x)):
        for artist in artists:
            artist.remove()

        ax.set_title('Cost: ' + str(cost[i]))
        artists = vg.plot_pose(ax, x[i].to_tuple(), scale=1)
        artists.extend(vg.utils.plot_as_box(ax, x[i] * points_o))
        plt.draw()
        while True:
            if plt.waitforbuttonpress():
                break


if __name__ == "__main__":
    main()
