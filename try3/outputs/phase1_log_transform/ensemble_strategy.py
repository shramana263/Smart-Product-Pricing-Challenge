
# Ensemble Strategy

# After training all 3 models, ensemble their predictions:

def ensemble_predictions(pred_log, pred_sqrt, pred_boxcox, weights=(0.5, 0.2, 0.3)):
    '''
    Combine predictions from 3 models with optimal weights
    
    Args:
        pred_log: Predictions from log-transformed model (already inverse-transformed)
        pred_sqrt: Predictions from sqrt-transformed model (already inverse-transformed)
        pred_boxcox: Predictions from boxcox model (already inverse-transformed)
        weights: Tuple of (w_log, w_sqrt, w_boxcox)
    
    Returns:
        final_predictions: Weighted ensemble
    '''
    w_log, w_sqrt, w_boxcox = weights
    final = w_log * pred_log + w_sqrt * pred_sqrt + w_boxcox * pred_boxcox
    return final

# Optimize weights on validation set
from scipy.optimize import minimize

def optimize_weights(pred_log, pred_sqrt, pred_boxcox, y_true):
    '''Find optimal ensemble weights'''
    
    def objective(weights):
        w_log, w_sqrt, w_boxcox = weights
        ensemble = w_log * pred_log + w_sqrt * pred_sqrt + w_boxcox * pred_boxcox
        return smape(y_true, ensemble)
    
    # Constraints: weights sum to 1, all non-negative
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1), (0, 1), (0, 1)]
    
    result = minimize(
        objective,
        x0=[0.5, 0.2, 0.3],  # Initial guess
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x

# Example usage:
# optimal_weights = optimize_weights(val_pred_log, val_pred_sqrt, val_pred_boxcox, val_y_true)
# final_pred = ensemble_predictions(test_pred_log, test_pred_sqrt, test_pred_boxcox, optimal_weights)
