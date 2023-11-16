# Tối ưu hóa ma trận xác suất chuyển
def optimize_transition_matrix(initial_transition_matrix, states, X, y):
    initial_transition_matrix_flat = np.array(initial_transition_matrix).flatten()

    # Constraint function to ensure probabilities sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x.reshape((len(states), len(states))), axis=1) - 1})
    
    # Bounds to ensure probabilities stay within [0, 1]
    bounds = [(0, 1) for _ in range(len(initial_transition_matrix_flat))]

    result = minimize(objective_function, initial_transition_matrix_flat, args=(states, X, y), method='SLSQP', constraints=constraints, bounds=bounds)
    
    # Normalize the transition matrix to ensure probabilities sum to 1
    normalized_transition_matrix = result.x.reshape((len(states), len(states)))
    normalized_transition_matrix /= normalized_transition_matrix.sum(axis=1, keepdims=True)

    return normalized_transition_matrix
