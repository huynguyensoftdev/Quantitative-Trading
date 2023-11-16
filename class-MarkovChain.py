class MarkovChain:
    def __init__(self, transition_matrix, states):
        self.transition_matrix = np.array(transition_matrix)
        self.states = states
        self.current_state = np.random.choice(states)

    def next_state(self):
        state_indices = np.arange(len(self.states))
        current_state_index = np.where(self.states == self.current_state)[0][0]
        
        # Normalize the transition probabilities to ensure they sum to 1
        transition_probabilities = self.transition_matrix[current_state_index]
        transition_probabilities /= transition_probabilities.sum()

        next_state_index = np.random.choice(state_indices, p=transition_probabilities)
        self.current_state = self.states[next_state_index]
        return self.current_state

    def set_transition_matrix(self, transition_matrix):
        self.transition_matrix = np.array(transition_matrix)
        # Normalize each row to ensure probabilities sum to 1
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
