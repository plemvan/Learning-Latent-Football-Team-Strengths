import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =================================================== #

class NeuralBradleyTerry(nn.Module):
    """
    Class implementing the Neural Bradley-Terry framework (NBTR).
    It learns a mapping from Team Features -> Team Strength.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, learning_rate: float = 0.01):
        """
        Initialize the Neural Network architecture.
        
        Parameters
        ----------
        input_dim : int
            Number of features per team (size of the feature vector).
        hidden_dim : int
            Size of the hidden layer.
        learning_rate : float
            Step size for the Adam optimizer.
        """
        super(NeuralBradleyTerry, self).__init__()
        
        # The "Score Function" f_phi(x)
        # Takes team features, outputs a scalar "strength"
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output is scalar (theta_i)
        )
        
        self.learning_rate = learning_rate
        # We use Adam, which is generally more stable for NNs than vanilla Gradient Descent
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # BCEWithLogitsLoss combines a Sigmoid layer and BCELoss in one single class
        # This is numerically more stable than using a plain Sigmoid followed by a BCELoss.
        # Loss = - [y * log(sigmoid(diff)) + (1-y) * log(1 - sigmoid(diff))]
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute strengths from features.
        """
        return self.network(x)

    def fit(self, X: np.array, features: np.array, n_iterations: int = 1000) -> 'NeuralBradleyTerry':
        """
        Fit the neural model to the data using implicit supervision.
        
        Parameters
        ----------
        X : np.array
            Adjacency matrix of results. X[i,j] = number of times i beat j.
        features : np.array
            Matrix of team features of shape (n_teams, n_features).
            Row i contains the stats for team i.
            
        Returns
        -------
        self : object
        """
        # Convert data to PyTorch tensors
        features_tensor = torch.FloatTensor(features)
        
        # 1. Pre-process X into pairs for training
        # We convert the matrix X into a list of matches: [(winner_idx, loser_idx), ...]
        # This is standard for training Neural Nets (batching).
        matches = []
        n_teams = X.shape[0]
        
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j: continue
                if X[i, j] > 0:
                    # If i beat j, we add this pair to the training set
                    # We can weigh this by the number of wins, or just repeat the entry
                    count = int(X[i, j])
                    for _ in range(count):
                        matches.append([i, j])
                        
        matches_tensor = torch.LongTensor(matches) # Shape (N_matches, 2)
        
        # Training Loop
        self.train() # Set mode to training
        for epoch in range(n_iterations):
            self.optimizer.zero_grad()
            
            # --- Step A: Compute Strengths for ALL teams ---
            # theta = f_phi(features)
            all_strengths = self.forward(features_tensor) # Shape (n_teams, 1)
            
            # --- Step B: Select strengths for the specific matches ---
            winners_idx = matches_tensor[:, 0]
            losers_idx = matches_tensor[:, 1]
            
            strength_winners = all_strengths[winners_idx]
            strength_losers  = all_strengths[losers_idx]
            
            # --- Step C: Compute Logits (Score Difference) ---
            # The model predicts P(i > j) based on (theta_i - theta_j)
            logits = strength_winners - strength_losers
            
            # --- Step D: Calculate Loss ---
            # The target is always 1.0 here because we constructed the list such that 
            # the first column (i) is the winner.
            targets = torch.ones_like(logits)
            
            loss = self.loss_fn(logits, targets)
            
            # --- Step E: Backpropagation ---
            loss.backward()
            self.optimizer.step()
            
            # Optional: Print loss every 100 iterations
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        return self

    def predict(self, features: np.array) -> np.array:
        """
        Return the predicted strengths for the given features.
        
        Parameters
        ----------
        features : np.array (n_teams, n_features)
        
        Returns
        -------
        theta : np.array
            Array of learned strengths.
        """
        self.eval() # Set mode to evaluation
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            strengths = self.forward(features_tensor)
        return strengths.numpy().flatten()