#========== Module for Gradient Descent ==========#

## Imports
import numpy as np

#=================================================#


class GradientDescent():

    """Class for Gradient Descent"""

    def __init__(self, learning_rate : float = 0.01, n_iterations : int = 1000, tolerance : float = 1e-6):
        
        """
        Gradient Descent algorithm

        Parameters
        ----------
        learning_rate : float. Default = 0.01
            Step of the gradient descent
        n_iterations : int. Default = 1000
            Number of iterations of the algorithm
        tolerance : float. Default = 1e-6
            Stop criterion (if the change is less than this threshold)
        """

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance

        return
    
    
    def optimize(self, gradient : callable, starting_point : np.array)->np.array:

        """
        Perform the gradient descent for the given function and its gradient

        Parameters
        ----------
        gradient : callable
            The gradient of the function we want to minimize
        starting_point : np.array
            The starting point for the descent
        
        Returns
        -------
        est_argmin : np.array
            The estimated argmin found by the algorithm
        """

        # Initialize the descent
        est_argmin = starting_point

        # Descent
        for i in range(self.n_iterations):
            
            # Computing gradient at the current point
            grad = gradient(est_argmin)

            # Stop criterion (if gradient norm is less than the threshold)
            if np.linalg.norm(grad) < self.tolerance:
                print(f"Tolerance threshold attained after {i} iterations")
                break
                
            # Updating current point
            est_argmin = est_argmin - self.learning_rate*grad

            if i == self.n_iterations-1:
                print(f"Algorithm terminates after {self.n_iterations} iterations")

        return est_argmin