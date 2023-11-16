# Hàm mục tiêu cần tối ưu
def objective_function(transition_matrix_flat, states, X, y):
    transition_matrix = transition_matrix_flat.reshape((len(states), len(states)))
    markov_model = MarkovChain(transition_matrix, states)

    markov_states_predictions = [markov_model.next_state() for _ in range(len(y))]
    mse = mean_squared_error(y, markov_states_predictions)

    return mse
