import numpy as np
def softmax(x):
    """Compute the softmax of each row of the input x.

    Args:
        x (np.ndarray): Input array of shape (n_samples, n_classes).

    Returns:
        np.ndarray: Softmax probabilities of shape (n_samples, n_classes).
    """
    # Subtract the max for numerical stability
    x_stable = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_stable)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)