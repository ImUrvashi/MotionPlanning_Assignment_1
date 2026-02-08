import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size=20):
        self.size = size
        self.grid = np.zeros((size, size))
        self.obstacles = []
        self.actions = ['up', 'down', 'left', 'right']

    def add_obstacle(self, x, y):
        """Adds an obstacle to the grid."""
        if 0 <= x < self.size and 0 <= y < self.size:
            self.grid[x, y] = 1  # 1 represents an obstacle
            self.obstacles.append((x, y))

    def get_state_from_pos(self, pos):
        """Converts a position (x, y) to a state index."""
        return pos[0] * self.size + pos[1]

    def get_pos_from_state(self, state):
        """Converts a state index to a position (x, y)."""
        return (state // self.size, state % self.size)

    def is_valid_pos(self, pos):
        """Checks if a position is within the grid and not an obstacle."""
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size and self.grid[x, y] == 0

    def get_next_state(self, state, action):
        """
        Returns the next state based on the stochastic transition model.
        - 0.8 probability of intended move
        - 0.1 probability of slip to each perpendicular direction
        """
        pos = self.get_pos_from_state(state)
        x, y = pos

        # Define action vectors
        action_vectors = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        # Define perpendicular slips
        perpendicular_slips = {
            'up': ['left', 'right'],
            'down': ['left', 'right'],
            'left': ['up', 'down'],
            'right': ['up', 'down']
        }

        # Get intended move
        intended_action = action
        slip_actions = perpendicular_slips[action]

        # Determine outcome based on probabilities
        rand_num = np.random.rand()
        if rand_num < 0.8:
            chosen_action = intended_action
        elif rand_num < 0.9:
            chosen_action = slip_actions[0]
        else:
            chosen_action = slip_actions[1]

        # Calculate next position
        dx, dy = action_vectors[chosen_action]
        next_pos = (x + dx, y + dy)

        # Check if the next position is valid
        if self.is_valid_pos(next_pos):
            return self.get_state_from_pos(next_pos)
        else:
            # If the move is invalid, stay in the current state
            return state

class PassiveADPAgent:
    def __init__(self, grid_world, goal_state):
        self.grid_world = grid_world
        self.goal_state = goal_state
        self.utilities = np.zeros(grid_world.size * grid_world.size)
        self.policy = self._create_policy()
        self.gamma = 0.9  # Discount factor
        # Store perpendicular slips in the agent
        self.perpendicular_slips = {
            'up': ['left', 'right'],
            'down': ['left', 'right'],
            'left': ['up', 'down'],
            'right': ['up', 'down']
        }


    def _create_policy(self):
        """Creates a simple policy to move towards the goal."""
        policy = {}
        for state in range(self.grid_world.size * self.grid_world.size):
            pos = self.grid_world.get_pos_from_state(state)
            goal_pos = self.grid_world.get_pos_from_state(self.goal_state)

            if pos == goal_pos:
                policy[state] = None
                continue

            dx = goal_pos[1] - pos[1]
            dy = goal_pos[0] - pos[0]

            if abs(dx) > abs(dy):
                if dx > 0:
                    policy[state] = 'right'
                else:
                    policy[state] = 'left'
            else:
                if dy > 0:
                    policy[state] = 'down'
                else:
                    policy[state] = 'up'
        return policy

    def get_action(self, state):
        """Returns the action from the policy."""
        return self.policy[state]

    def update_utilities(self):
        """Performs one iteration of policy evaluation."""
        new_utilities = self.utilities.copy()
        for state in range(self.grid_world.size * self.grid_world.size):
            if self.grid_world.is_valid_pos(self.grid_world.get_pos_from_state(state)):
                action = self.policy[state]
                if action:
                    expected_utility = 0
                    # Intended move
                    next_state = self.grid_world.get_next_state(state, action)
                    reward = -1 if next_state != self.goal_state else 100
                    expected_utility += 0.8 * (reward + self.gamma * self.utilities[next_state])

                    # Slip moves
                    slip_actions = self.perpendicular_slips[action]
                    for slip_action in slip_actions:
                        next_state = self.grid_world.get_next_state(state, slip_action)
                        reward = -1 if next_state != self.goal_state else 100
                        expected_utility += 0.1 * (reward + self.gamma * self.utilities[next_state])
                    new_utilities[state] = expected_utility
        self.utilities = new_utilities

    def estimate_transition_matrix(self, episodes=1000):
        """Estimates the transition matrix from experience."""
        num_states = self.grid_world.size * self.grid_world.size
        num_actions = len(self.grid_world.actions)
        counts = np.zeros((num_states, num_actions, num_states))

        for _ in range(episodes):
            state = np.random.randint(num_states)
            if self.grid_world.is_valid_pos(self.grid_world.get_pos_from_state(state)):
                action_index = np.random.randint(len(self.grid_world.actions))
                action = self.grid_world.actions[action_index]
                next_state = self.grid_world.get_next_state(state, action)
                counts[state, action_index, next_state] += 1
        
        # To avoid division by zero
        counts[counts.sum(axis=2) == 0] = 1
        
        return counts / counts.sum(axis=2, keepdims=True)

class QLearningAgent:
    def __init__(self, grid_world, goal_state, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.grid_world = grid_world
        self.goal_state = goal_state
        self.q_table = np.zeros((grid_world.size * grid_world.size, len(grid_world.actions)))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def get_action(self, state):
        """
        Returns an action using an epsilon-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return np.random.choice(self.grid_world.actions)
        else:
            # Exploit: choose the best action from the Q-table
            return self.grid_world.actions[np.argmax(self.q_table[state])]

    def update_q_value(self, state, action, reward, next_state):
        """
        Updates the Q-value for a state-action pair.
        """
        action_index = self.grid_world.actions.index(action)
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_delta = td_target - self.q_table[state, action_index]
        self.q_table[state, action_index] += self.alpha * td_delta

def get_true_transition_matrix(grid_world):
    """Computes the true transition matrix."""
    num_states = grid_world.size * grid_world.size
    num_actions = len(grid_world.actions)
    transition_matrix = np.zeros((num_states, num_actions, num_states))
    action_vectors = {
        'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)
    }
    perpendicular_slips = {
        'up': ['left', 'right'], 'down': ['left', 'right'],
        'left': ['up', 'down'], 'right': ['up', 'down']
    }

    for state in range(num_states):
        if grid_world.is_valid_pos(grid_world.get_pos_from_state(state)):
            for action_index, action in enumerate(grid_world.actions):
                pos = grid_world.get_pos_from_state(state)
                x,y = pos

                # Intended move
                dx, dy = action_vectors[action]
                next_pos = (x + dx, y + dy)
                next_state = grid_world.get_state_from_pos(next_pos) if grid_world.is_valid_pos(next_pos) else state
                transition_matrix[state, action_index, next_state] += 0.8
                
                # Slip moves
                slip_actions = perpendicular_slips[action]
                for slip_action in slip_actions:
                    dx, dy = action_vectors[slip_action]
                    next_pos = (x + dx, y + dy)
                    next_state = grid_world.get_state_from_pos(next_pos) if grid_world.is_valid_pos(next_pos) else state
                    transition_matrix[state, action_index, next_state] += 0.1
    return transition_matrix

def main():
    # Create grid world
    grid_world = GridWorld(size=20)

    # Add some obstacles
    for _ in range(80): # 20% obstacles
        x, y = np.random.randint(0, 20, size=2)
        grid_world.add_obstacle(x, y)

    # Generate 10 unique source-destination pairs
    source_dest_pairs = []
    while len(source_dest_pairs) < 10:
        source = (np.random.randint(0, 20), np.random.randint(0, 20))
        dest = (np.random.randint(0, 20), np.random.randint(0, 20))
        if source != dest and grid_world.is_valid_pos(source) and grid_world.is_valid_pos(dest):
            source_dest_pairs.append((source, dest))

    # --- Simulation ---
    all_trajectories = {'adp': [], 'q_learning': []}
    source_state = grid_world.get_state_from_pos(source_dest_pairs[0][0])
    dest_state = grid_world.get_state_from_pos(source_dest_pairs[0][1])

    for source_pos, dest_pos in source_dest_pairs:
        source_state_loop = grid_world.get_state_from_pos(source_pos)
        dest_state_loop = grid_world.get_state_from_pos(dest_pos)

        # --- Passive ADP Agent ---
        adp_agent = PassiveADPAgent(grid_world, dest_state_loop)
        # Train ADP agent
        for _ in range(100): # Convergence iterations
            adp_agent.update_utilities()

        # Generate trajectories for ADP agent
        for _ in range(3):
            trajectory = []
            current_state = source_state_loop
            while current_state != dest_state_loop and len(trajectory) < 100:
                action = adp_agent.get_action(current_state)
                if action is None: # Goal state has no action
                    break
                next_state = grid_world.get_next_state(current_state, action)
                trajectory.append(grid_world.get_pos_from_state(current_state))
                current_state = next_state
            if current_state == dest_state_loop:
                trajectory.append(grid_world.get_pos_from_state(current_state))
            all_trajectories['adp'].append(trajectory)


        # --- Active Q-Learning Agent ---
        q_agent = QLearningAgent(grid_world, dest_state_loop)
        # Train Q-learning agent
        for _ in range(1000): # Training episodes
            current_state = source_state_loop
            while current_state != dest_state_loop:
                action = q_agent.get_action(current_state)
                next_state = grid_world.get_next_state(current_state, action)
                reward = -1 if next_state != dest_state_loop else 100
                q_agent.update_q_value(current_state, action, reward, next_state)
                current_state = next_state

        # Generate trajectories for Q-learning agent
        for _ in range(3):
            trajectory = []
            current_state = source_state_loop
            while current_state != dest_state_loop and len(trajectory) < 100:
                action = q_agent.get_action(current_state)
                next_state = grid_world.get_next_state(current_state, action)
                trajectory.append(grid_world.get_pos_from_state(current_state))
                current_state = next_state
            if current_state == dest_state_loop:
                trajectory.append(grid_world.get_pos_from_state(current_state))
            all_trajectories['q_learning'].append(trajectory)


    # --- Visualization ---
    def visualize_trajectories(grid_world, trajectories, title):
        plt.figure(figsize=(10, 10))
        # Plot grid and obstacles
        plt.imshow(grid_world.grid, cmap='Greys', origin='lower')
        # Plot trajectories
        for trajectory in trajectories:
            if trajectory:
                path = np.array(trajectory)
                plt.plot(path[:, 1], path[:, 0], marker='o', markersize=3)
        # Mark start and end points
        for source, dest in source_dest_pairs:
            plt.plot(source[1], source[0], 'go', markersize=10, label='Start')
            plt.plot(dest[1], dest[0], 'ro', markersize=10, label='End')
        plt.title(title)
        plt.show()

    visualize_trajectories(grid_world, all_trajectories['adp'], 'ADP Agent Trajectories')
    visualize_trajectories(grid_world, all_trajectories['q_learning'], 'Q-Learning Agent Trajectories')

    # --- Analysis ---
    adp_path_lengths = [len(t) for t in all_trajectories['adp']]
    q_learning_path_lengths = [len(t) for t in all_trajectories['q_learning']]

    print("--- Path Length Analysis ---")
    print(f"ADP Agent Average Path Length: {np.mean(adp_path_lengths):.2f}")
    print(f"Q-Learning Agent Average Path Length: {np.mean(q_learning_path_lengths):.2f}")

    # --- Convergence Analysis ---
    adp_utility_changes = []
    adp_agent_for_conv = PassiveADPAgent(grid_world, dest_state)
    for _ in range(100):
        old_utilities = adp_agent_for_conv.utilities.copy()
        adp_agent_for_conv.update_utilities()
        adp_utility_changes.append(np.sum(np.abs(adp_agent_for_conv.utilities - old_utilities)))

    q_value_changes = []
    q_agent_for_conv = QLearningAgent(grid_world, dest_state)
    for i in range(1000):
        old_q_table = q_agent_for_conv.q_table.copy()
        current_state = source_state
        while current_state != dest_state:
            action = q_agent_for_conv.get_action(current_state)
            next_state = grid_world.get_next_state(current_state, action)
            reward = -1 if next_state != dest_state else 100
            q_agent_for_conv.update_q_value(current_state, action, reward, next_state)
            current_state = next_state
        if i % 10 == 0:
            q_value_changes.append(np.sum(np.abs(q_agent_for_conv.q_table - old_q_table)))


    plt.figure(figsize=(10, 5))
    plt.plot(adp_utility_changes, label='ADP Utility Change')
    plt.xlabel('Iteration')
    plt.ylabel('Sum of Absolute Utility Change')
    plt.title('ADP Agent Convergence')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(q_value_changes, label='Q-value Change')
    plt.xlabel('Episode (x10)')
    plt.ylabel('Sum of Absolute Q-value Change')
    plt.title('Q-Learning Agent Convergence')
    plt.legend()
    plt.show()
    
    # --- Transition Matrix Estimation ---
    adp_agent_for_t = PassiveADPAgent(grid_world, dest_state)
    estimated_T = adp_agent_for_t.estimate_transition_matrix()
    true_T = get_true_transition_matrix(grid_world)

    # Compute error
    error = np.sum(np.abs(estimated_T - true_T))
    print(f"Transition Matrix Estimation Error (Sum of Absolute Differences): {error:.2f}")

    # Visualize the difference for a specific state and action
    state_to_visualize = source_state
    action_index_to_visualize = 0 # 'up'
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(true_T[state_to_visualize, action_index_to_visualize, :].reshape(grid_world.size, grid_world.size), cmap='hot', interpolation='nearest')
    plt.title("True Transition Probabilities")
    plt.subplot(1, 2, 2)
    plt.imshow(estimated_T[state_to_visualize, action_index_to_visualize, :].reshape(grid_world.size, grid_world.size), cmap='hot', interpolation='nearest')
    plt.title("Estimated Transition Probabilities")
    plt.show()


# if __name__ == '__main__':
    main()
