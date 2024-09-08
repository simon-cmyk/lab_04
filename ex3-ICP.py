from scipy.spatial import KDTree

import numpy as np
import visgeom as vg
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
from pylie import SO3, SE3
from optim import gauss_newton, Levenberg_Marquart

"""Example 3 - ICP estimation"""


def generate_dense_box(num_points_per_edge=100):
    """Generate a dense point cloud for a 3D box."""
  
    x = np.linspace(-1, 1, num_points_per_edge)
    y = np.linspace(-1, 1, num_points_per_edge)
    z = np.linspace(-1, 1, num_points_per_edge)
    
    # Create meshgrid to get all combinations of points on the surface
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Flatten the grid and concatenate to get the 3D point cloud
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

    return points

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
            print("Using prior")

        return A, b, b.T.dot(b)

class ICPBasedPoseEstimatorObjective(NoisyPointAlignmentBasedPoseEstimatorObjective):
    """Implements ICP strategy for dynamic point correspondences"""

    def __init__(self, x_w, x_o, covs, T_prior=None, Cov_prior=None):
        super().__init__(x_w, x_o, covs, T_prior, Cov_prior)

    def find_correspondences(self, T_wo):
        """Update the correspondences using ICP by finding closest points"""
        # Transform observed points using the current pose estimate
        transformed_points_o = T_wo.inverse() * self.x_w

        # Use a KDTree for fast closest point lookup
        kdtree = KDTree(transformed_points_o.T)  # Create a KDTree for the observed points
        
        # Find closest points in the "world" point cloud for each point in "observed" point cloud
        distances, indices = kdtree.query(self.x_o.T)  # Query the KDTree

        # Update the point correspondences
        self.x_w = self.x_w[:, indices]
        return distances

def main():
    # Generate a point cloud (e.g., a box or another shape)
    points_w = generate_dense_box(num_points_per_edge=10)

    # True observer pose (ground truth)
    true_pose_wo = SE3((SO3.rot_z(np.pi / 4), np.array([[3, 3, 0]]).T))

    # Apply transformation to points to simulate the observed point cloud
    points_o = true_pose_wo.inverse() * points_w
    num_points = points_o.shape[1]

    # Add noise to observed points
    point_covariances = [np.diag(np.array([1e-1, 1e-1, 1e-1]) ** 2)] * num_points
    for c in range(num_points):
        points_o[:, [c]] = points_o[:, [c]] + np.random.multivariate_normal(np.zeros(3), point_covariances[c]).reshape(
            -1, 1)

    # Perturb the observer pose for initial guess
    init_pose_wo = true_pose_wo + 1e-1 * np.random.randn(6, 1)

    # Use ICP to dynamically update correspondences and estimate the pose
    model = ICPBasedPoseEstimatorObjective(points_w, points_o, point_covariances)

    # Iterate with ICP: find correspondences, estimate pose, and repeat
    max_iterations = 50
    tolerance = 1e-5
    prev_error = float('inf')

    for it in range(max_iterations):
        # Step 1: Find correspondences (update model correspondences)
        distances = model.find_correspondences(init_pose_wo)
        
        # Step 2: Estimate pose using the updated correspondences
        x, cost, A, b = gauss_newton(init_pose_wo, model, cost_thresh=1e-14, delta_thresh=1e-14, max_num_it=14)  # Single GN iteration

        # Step 3: Check convergence
        error = sum(distances)
        print(f"Iteration {it}, error: {error}")
        
        # Update the pose for the next iteration
        init_pose_wo = x[-1]  # Use the latest pose estimate for the next iteration

    # Final pose estimate and error
    print(f"Final estimated pose:\n{init_pose_wo}")
    print(f"True pose:\n{true_pose_wo}")

    # Visualize results
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot true and estimated poses
    vg.plot_pose(ax, true_pose_wo.to_tuple(), scale=1, alpha=0.4, color='green')
    vg.plot_pose(ax, init_pose_wo.to_tuple(), scale=1, alpha=0.4, color='red')

    # Plot world and observed point clouds
    vg.utils.plot_as_box(ax, points_w, color='blue')
    vg.utils.plot_as_box(ax, points_o, color='red')

    plt.show()

if __name__ == "__main__":
    main()