import numpy as np

# Define the grid world dimensions
grid_rows = 3
grid_cols = 3

# Define rewards for different states
rewards = np.zeros((grid_rows, grid_cols))
rewards[2, 2] = 1  # Reward for reaching the goal state

# Define obstacle positions (represented as -1)
obstacles = [(1, 1)]

# Define discount factor
gamma = 0.9


# Define a function to calculate the next state and reward based on the current state and action
def get_next_state_and_reward(state, action):
    if state == (2, 2):  # Goal state
        return (2, 2), 0

    row, col = state
    if action == "up":
        row = max(0, row - 1)
    elif action == "down":
        row = min(grid_rows - 1, row + 1)
    elif action == "left":
        col = max(0, col - 1)
    elif action == "right":
        col = min(grid_cols - 1, col + 1)

    if (row, col) in obstacles:
        return state, -0.1  # Penalty for hitting an obstacle

    return (row, col), rewards[row, col]


# Initialize the value function
value_function = np.zeros((grid_rows, grid_cols))

# Perform value iteration
num_iterations = 100
for _ in range(num_iterations):
    new_value_function = np.zeros((grid_rows, grid_cols))

    for row in range(grid_rows):
        for col in range(grid_cols):
            if (row, col) == (2, 2):  # Goal state
                new_value_function[row, col] = 1
            else:
                max_value = -float("inf")
                for action in ["up", "down", "left", "right"]:
                    next_state, reward = get_next_state_and_reward((row, col), action)
                    expected_value = reward + gamma * value_function[next_state[0], next_state[1]]
                    max_value = max(max_value, expected_value)
                new_value_function[row, col] = max_value

    value_function = new_value_function

# Print the learned value function
print("Learned Value Function:")
print(value_function)

# Extract the optimal policy
optimal_policy = np.zeros((grid_rows, grid_cols), dtype=str)
for row in range(grid_rows):
    for col in range(grid_cols):
        if (row, col) == (2, 2):  # Goal state
            optimal_policy[row, col] = "Goal"
        else:
            max_action = None
            max_value = -float("inf")
            for action in ["up", "down", "left", "right"]:
                next_state, _ = get_next_state_and_reward((row, col), action)
                expected_value = gamma * value_function[next_state[0], next_state[1]]
                if expected_value > max_value:
                    max_value = expected_value
                    max_action = action
            optimal_policy[row, col] = max_action

# Print the optimal policy
print("\nOptimal Policy:")
print(optimal_policy)
