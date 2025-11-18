import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random


# WINDY GRIDWORLD ENVIRONMENT


class WindyGridworld:
    """
    Stochastic Windy Gridworld Environment
    Based on the assignment specifications
    """
    
    def __init__(self, height=7, width=10, start=(3, 0), goal=(3, 7), wind_strengths=None):
        self.height = height
        self.width = width
        self.start_state = start
        self.goal_state = goal
        
        # Wind pattern for each column
        if wind_strengths is None:
            self.wind_strengths = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        else:
            self.wind_strengths = wind_strengths
        
        # Action definitions: up, down, left, right
        self.action_space = [0, 1, 2, 3]
        self.action_effects = {
            0: (-1, 0),   # up
            1: (1, 0),    # down
            2: (0, -1),   # left
            3: (0, 1)     # right
        }
        
        self.current_state = self.start_state
    
    def reset_environment(self):
        """Reset to starting position"""
        self.current_state = self.start_state
        return self.current_state
    
    def is_terminal_state(self, state):
        """Check if state is terminal (goal)"""
        return state == self.goal_state
    
    def get_wind_force(self, column):
        """
        Get stochastic wind force for a column
        10% chance: wind + 1
        80% chance: normal wind  
        10% chance: no wind
        """
        random_val = random.random()
        if random_val < 0.1:
            return self.wind_strengths[column] + 1
        elif random_val < 0.9:
            return self.wind_strengths[column]
        else:
            return 0
    
    def transition(self, action):
        """
        Execute action and return (next_state, reward, done)
        """
        if self.is_terminal_state(self.current_state):
            return self.current_state, 0, True
        
        current_row, current_col = self.current_state
        
        # Step 1: Apply the chosen action
        row_change, col_change = self.action_effects[action]
        new_row, new_col = current_row + row_change, current_col + col_change
        
        # Step 2: Apply stochastic wind
        wind_power = self.get_wind_force(current_col)
        wind_affected_row = new_row - wind_power
        wind_affected_col = new_col
        
        # Step 3: Ensure within grid boundaries
        final_row = max(0, min(wind_affected_row, self.height - 1))
        final_col = max(0, min(wind_affected_col, self.width - 1))
        
        next_state = (final_row, final_col)
        
        # Reward: 0 for reaching goal, -1 otherwise
        reward = 0 if next_state == self.goal_state else -1
        episode_done = self.is_terminal_state(next_state)
        
        self.current_state = next_state
        return next_state, reward, episode_done
    
    def get_all_possible_states(self):
        """Get all non-terminal states"""
        states_list = []
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) != self.goal_state:
                    states_list.append((row, col))
        return states_list

# POLICY FUNCTIONS


def get_epsilon_greedy_action(Q_function, state, available_actions, epsilon_val):
    """
    Epsilon-greedy action selection
    """
    if random.random() < epsilon_val:
        return random.choice(available_actions)
    else:
        # Get Q-values for all actions in current state
        action_values = [Q_function.get((state, action), 0.0) for action in available_actions]
        best_value = max(action_values)
        # Handle multiple actions with same Q-value
        best_actions = [action for action in available_actions 
                       if Q_function.get((state, action), 0.0) == best_value]
        return random.choice(best_actions)

def get_greedy_action(Q_function, state, available_actions):
    """
    Greedy action selection
    """
    action_values = [Q_function.get((state, action), 0.0) for action in available_actions]
    best_value = max(action_values)
    best_actions = [action for action in available_actions 
                   if Q_function.get((state, action), 0.0) == best_value]
    return random.choice(best_actions)

def generate_episode_record(env, policy_func, Q_dict=None, epsilon_val=0.1):
    """
    Generate a complete episode following policy
    Returns: list of (state, action, reward) tuples
    """
    episode_history = []
    current_state = env.reset_environment()
    
    while True:
        if Q_dict is not None:
            chosen_action = policy_func(Q_dict, current_state, env.action_space, epsilon_val)
        else:
            chosen_action = policy_func(current_state, env.action_space)
        
        next_state, immediate_reward, done_flag = env.transition(chosen_action)
        episode_history.append((current_state, chosen_action, immediate_reward))
        
        if done_flag:
            break
        current_state = next_state
    
    return episode_history


# ALGORITHM 1: DYNAMIC PROGRAMMING CONTROL


def dynamic_programming_control(environment, discount_factor=1.0, convergence_threshold=1e-6, max_iterations=1000):
    """
    Dynamic Programming Control using Value Iteration
    Requires full knowledge of environment dynamics
    """
    print("Running Dynamic Programming Control...")
    
    # Initialize value function
    state_values = defaultdict(float)
    policy_dict = {}
    
    # Get all states including terminal
    all_states = environment.get_all_possible_states()
    all_states.append(environment.goal_state)
    
    for iteration_count in range(max_iterations):
        max_delta = 0
        
        # Update each state's value
        for state in all_states:
            if environment.is_terminal_state(state):
                state_values[state] = 0
                continue
            
            # Calculate Q-values for all possible actions
            q_values_list = []
            for action in environment.action_space:
                q_value = 0
                # Consider all wind outcomes with their probabilities
                wind_probabilities = [0.1, 0.8, 0.1]
                wind_variations = [1, 0, -1]
                
                current_row, current_col = state
                row_delta, col_delta = environment.action_effects[action]
                
                for prob, wind_mod in zip(wind_probabilities, wind_variations):
                    # Determine wind strength for this outcome
                    if wind_mod == 1:
                        actual_wind = environment.wind_strengths[current_col] + 1
                    elif wind_mod == 0:
                        actual_wind = environment.wind_strengths[current_col]
                    else:
                        actual_wind = 0
                    
                    # Calculate resulting state
                    result_row = current_row + row_delta - actual_wind
                    result_col = current_col + col_delta
                    
                    # Apply boundary constraints
                    final_row = max(0, min(result_row, environment.height - 1))
                    final_col = max(0, min(result_col, environment.width - 1))
                    
                    resulting_state = (final_row, final_col)
                    reward_val = 0 if resulting_state == environment.goal_state else -1
                    
                    q_value += prob * (reward_val + discount_factor * state_values[resulting_state])
                
                q_values_list.append(q_value)
            
            new_value = max(q_values_list)
            max_delta = max(max_delta, abs(state_values[state] - new_value))
            state_values[state] = new_value
        
        if max_delta < convergence_threshold:
            print(f"  DP converged after {iteration_count + 1} iterations")
            break
    
    # Extract Q-function from value function
    Q_function = defaultdict(float)
    for state in all_states:
        if environment.is_terminal_state(state):
            continue
        
        for action in environment.action_space:
            q_val = 0
            wind_probs = [0.1, 0.8, 0.1]
            wind_changes = [1, 0, -1]
            
            current_row, current_col = state
            row_delta, col_delta = environment.action_effects[action]
            
            for prob, wind_change in zip(wind_probs, wind_changes):
                if wind_change == 1:
                    wind_strength = environment.wind_strengths[current_col] + 1
                elif wind_change == 0:
                    wind_strength = environment.wind_strengths[current_col]
                else:
                    wind_strength = 0
                
                new_row = current_row + row_delta - wind_strength
                new_col = current_col + col_delta
                
                final_row = max(0, min(new_row, environment.height - 1))
                final_col = max(0, min(new_col, environment.width - 1))
                
                next_state = (final_row, final_col)
                reward_val = 0 if next_state == environment.goal_state else -1
                
                q_val += prob * (reward_val + discount_factor * state_values[next_state])
            
            Q_function[(state, action)] = q_val
    
    return Q_function, state_values


# ALGORITHM 2: MONTE CARLO ON-POLICY CONTROL


def monte_carlo_on_policy(environment, total_episodes=5000, discount_factor=1.0, epsilon_val=0.1, learning_rate=0.1):
    """
    Monte Carlo On-Policy Control with epsilon-greedy policy
    """
    print("Running Monte Carlo On-Policy Control...")
    
    Q_function = defaultdict(float)
    visit_counts = defaultdict(int)
    episode_step_counts = []
    
    for episode_index in range(total_episodes):
        # Generate episode using current policy
        episode_data = generate_episode_record(environment, get_epsilon_greedy_action, Q_function, epsilon_val)
        episode_step_counts.append(len(episode_data))
        
        # Track visited state-action pairs (first-visit)
        visited_pairs = set()
        
        # Calculate returns and update Q-values
        cumulative_return = 0
        for step in range(len(episode_data) - 1, -1, -1):
            state, action, reward = episode_data[step]
            cumulative_return = discount_factor * cumulative_return + reward
            
            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                # Incremental update of Q-values
                visit_counts[(state, action)] += 1
                current_q = Q_function[(state, action)]
                Q_function[(state, action)] = current_q + learning_rate * (cumulative_return - current_q)
        
        if (episode_index + 1) % 500 == 0:
            avg_steps = np.mean(episode_step_counts[-500:])
            print(f"  Episode {episode_index + 1}/{total_episodes}, Average Steps: {avg_steps:.2f}")
    
    return Q_function, episode_step_counts


# ALGORITHM 3: MONTE CARLO OFF-POLICY CONTROL (UNWEIGHTED)


def monte_carlo_off_policy_unweighted(environment, total_episodes=5000, discount_factor=1.0, epsilon_val=0.1, learning_rate=0.1):
    """
    Monte Carlo Off-Policy Control with Unweighted Importance Sampling
    """
    print("Running Monte Carlo Off-Policy Control (Unweighted)...")
    
    Q_function = defaultdict(float)
    episode_step_counts = []
    
    for episode_index in range(total_episodes):
        # Generate episode using behavior policy
        episode_data = generate_episode_record(environment, get_epsilon_greedy_action, Q_function, epsilon_val)
        episode_step_counts.append(len(episode_data))
        
        # Track visited state-action pairs
        visited_pairs = set()
        cumulative_return = 0
        
        # Calculate importance sampling ratio
        importance_ratio = 1.0
        
        for step in range(len(episode_data) - 1, -1, -1):
            state, action, reward = episode_data[step]
            cumulative_return = discount_factor * cumulative_return + reward
            
            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                # Unweighted importance sampling update
                current_q = Q_function[(state, action)]
                Q_function[(state, action)] = current_q + learning_rate * importance_ratio * (cumulative_return - current_q)
            
            # Update importance sampling ratio
            greedy_act = get_greedy_action(Q_function, state, environment.action_space)
            if action == greedy_act:
                target_prob = 1.0
                behavior_prob = epsilon_val / len(environment.action_space) + (1 - epsilon_val)
            else:
                target_prob = 0.0
                behavior_prob = epsilon_val / len(environment.action_space)
            
            if target_prob == 0:
                break
            
            importance_ratio *= target_prob / behavior_prob
        
        if (episode_index + 1) % 500 == 0:
            avg_steps = np.mean(episode_step_counts[-500:])
            print(f"  Episode {episode_index + 1}/{total_episodes}, Average Steps: {avg_steps:.2f}")
    
    return Q_function, episode_step_counts


# ALGORITHM 4: MONTE CARLO OFF-POLICY CONTROL (WEIGHTED)


def monte_carlo_off_policy_weighted(environment, total_episodes=5000, discount_factor=1.0, epsilon_val=0.1, learning_rate=0.1):
    """
    Monte Carlo Off-Policy Control with Weighted Importance Sampling
    """
    print("Running Monte Carlo Off-Policy Control (Weighted)...")
    
    Q_function = defaultdict(float)
    weight_sums = defaultdict(float)
    episode_step_counts = []
    
    for episode_index in range(total_episodes):
        # Generate episode using behavior policy
        episode_data = generate_episode_record(environment, get_epsilon_greedy_action, Q_function, epsilon_val)
        episode_step_counts.append(len(episode_data))
        
        cumulative_return = 0
        importance_ratio = 1.0
        
        for step in range(len(episode_data) - 1, -1, -1):
            state, action, reward = episode_data[step]
            cumulative_return = discount_factor * cumulative_return + reward
            
            # Update weights and Q-values
            weight_sums[(state, action)] = weight_sums.get((state, action), 0.0) + importance_ratio
            current_q = Q_function.get((state, action), 0.0)
            Q_function[(state, action)] = current_q + (importance_ratio / weight_sums[(state, action)]) * (cumulative_return - current_q)
            
            # Update importance sampling ratio
            greedy_act = get_greedy_action(Q_function, state, environment.action_space)
            if action != greedy_act:
                break
            
            # Calculate probabilities
            if action == greedy_act:
                target_prob = 1.0
                behavior_prob = epsilon_val / len(environment.action_space) + (1 - epsilon_val)
            else:
                target_prob = 0.0
                behavior_prob = epsilon_val / len(environment.action_space)
            
            importance_ratio *= target_prob / behavior_prob
        
        if (episode_index + 1) % 500 == 0:
            avg_steps = np.mean(episode_step_counts[-500:])
            print(f"  Episode {episode_index + 1}/{total_episodes}, Average Steps: {avg_steps:.2f}")
    
    return Q_function, episode_step_counts


# ALGORITHM 5: TD(0) ON-POLICY CONTROL (SARSA)


def td_sarsa_control(environment, total_episodes=5000, discount_factor=1.0, epsilon_val=0.1, learning_rate=0.1):
    """
    TD(0) On-Policy Control (SARSA)
    """
    print("Running TD(0) On-Policy Control (SARSA)...")
    
    Q_function = defaultdict(float)
    episode_step_counts = []
    
    for episode_index in range(total_episodes):
        current_state = environment.reset_environment()
        current_action = get_epsilon_greedy_action(Q_function, current_state, environment.action_space, epsilon_val)
        
        step_count = 0
        while True:
            next_state, reward, done = environment.transition(current_action)
            step_count += 1
            
            if done:
                # Update Q-value for terminal transition
                Q_function[(current_state, current_action)] += learning_rate * (reward - Q_function[(current_state, current_action)])
                break
            
            # Choose next action using policy
            next_action = get_epsilon_greedy_action(Q_function, next_state, environment.action_space, epsilon_val)
            
            # SARSA update
            td_target = reward + discount_factor * Q_function[(next_state, next_action)]
            td_error = td_target - Q_function[(current_state, current_action)]
            Q_function[(current_state, current_action)] += learning_rate * td_error
            
            current_state = next_state
            current_action = next_action
        
        episode_step_counts.append(step_count)
        
        if (episode_index + 1) % 500 == 0:
            avg_steps = np.mean(episode_step_counts[-500:])
            print(f"  Episode {episode_index + 1}/{total_episodes}, Average Steps: {avg_steps:.2f}")
    
    return Q_function, episode_step_counts


# ALGORITHM 6: TD(0) OFF-POLICY CONTROL (Q-LEARNING)


def td_q_learning_control(environment, total_episodes=5000, discount_factor=1.0, epsilon_val=0.1, learning_rate=0.1):
    """
    TD(0) Off-Policy Control (Q-Learning)
    """
    print("Running TD(0) Off-Policy Control (Q-Learning)...")
    
    Q_function = defaultdict(float)
    episode_step_counts = []
    
    for episode_index in range(total_episodes):
        current_state = environment.reset_environment()
        
        step_count = 0
        while True:
            # Choose action using behavior policy (epsilon-greedy)
            current_action = get_epsilon_greedy_action(Q_function, current_state, environment.action_space, epsilon_val)
            next_state, reward, done = environment.transition(current_action)
            step_count += 1
            
            if done:
                Q_function[(current_state, current_action)] += learning_rate * (reward - Q_function[(current_state, current_action)])
                break
            
            # Q-Learning update (off-policy)
            max_next_q = max([Q_function.get((next_state, action), 0.0) for action in environment.action_space])
            td_target = reward + discount_factor * max_next_q
            td_error = td_target - Q_function[(current_state, current_action)]
            Q_function[(current_state, current_action)] += learning_rate * td_error
            
            current_state = next_state
        
        episode_step_counts.append(step_count)
        
        if (episode_index + 1) % 500 == 0:
            avg_steps = np.mean(episode_step_counts[-500:])
            print(f"  Episode {episode_index + 1}/{total_episodes}, Average Steps: {avg_steps:.2f}")
    
    return Q_function, episode_step_counts


# VISUALIZATION AND ANALYSIS


def plot_algorithm_comparison(results_data, title="Windy Gridworld: Algorithm Performance"):
    """
    Plot learning curves for all algorithms
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, (algorithm_name, step_data) in enumerate(results_data.items()):
        # Apply moving average for smoothing
        window_size = 100
        if len(step_data) > window_size:
            smoothed_data = np.convolve(step_data, np.ones(window_size)/window_size, mode='valid')
            episodes = range(len(smoothed_data))
            plt.plot(episodes, smoothed_data, label=algorithm_name, color=colors[idx % len(colors)], linewidth=2)
        else:
            plt.plot(step_data, label=algorithm_name, color=colors[idx % len(colors)], linewidth=2)
    
    plt.xlabel('Episode Number', fontsize=14)
    plt.ylabel('Steps per Episode', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def display_learned_strategy(environment, Q_function):
    """
    Display the learned policy visually
    """
    action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    policy_grid = [['' for _ in range(environment.width)] for _ in range(environment.height)]
    
    for row in range(environment.height):
        for col in range(environment.width):
            state = (row, col)
            if state == environment.goal_state:
                policy_grid[row][col] = 'G'
            elif state == environment.start_state:
                q_vals = [Q_function.get((state, action), 0.0) for action in environment.action_space]
                best_act = np.argmax(q_vals)
                policy_grid[row][col] = 'S' + action_symbols[best_act]
            else:
                q_vals = [Q_function.get((state, action), 0.0) for action in environment.action_space]
                best_act = np.argmax(q_vals)
                policy_grid[row][col] = action_symbols[best_act]
    
    print("\nLearned Policy Visualization:")
    print("S = Start, G = Goal")
    print("Wind pattern by column:", environment.wind_strengths)
    for row in policy_grid:
        print(' '.join(f'{cell:>3}' for cell in row))
    print()


# MAIN EXPERIMENT


def run_comprehensive_experiment():
    """
    Run all algorithms and compare their performance
    """
    print("=" * 70)
    print("WINDY GRIDWORLD - REINFORCEMENT LEARNING EXPERIMENT")
    print("=" * 70)
    
    # Create the Windy Gridworld environment
    grid_env = WindyGridworld(
        height=7,
        width=10,
        start=(3, 0),
        goal=(3, 7),
        wind_strengths=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    )
    
    print(f"\nEnvironment Configuration:")
    print(f"  Grid Size: {grid_env.height} x {grid_env.width}")
    print(f"  Start Position: {grid_env.start_state}")
    print(f"  Goal Position: {grid_env.goal_state}")
    print(f"  Wind Strengths: {grid_env.wind_strengths}")
    print()
    
    # Algorithm parameters
    num_episodes = 3000
    discount_factor = 1.0
    epsilon_val = 0.1
    learning_rate = 0.1
    
    results_storage = {}
    Q_functions = {}
    
    # Execute all required algorithms
    print("\n" + "=" * 70)
    print("EXECUTING CONTROL ALGORITHMS")
    print("=" * 70 + "\n")
    
    # 1. Dynamic Programming Control - FIXED: using correct parameter name
    Q_dp, V_dp = dynamic_programming_control(grid_env, discount_factor=discount_factor)
    Q_functions['DP Control'] = Q_dp
    print()
    
    # 2. Monte Carlo On-Policy Control
    Q_mc_on, steps_mc_on = monte_carlo_on_policy(
        grid_env, total_episodes=num_episodes, discount_factor=discount_factor, 
        epsilon_val=epsilon_val, learning_rate=learning_rate
    )
    results_storage['MC On-Policy'] = steps_mc_on
    Q_functions['MC On-Policy'] = Q_mc_on
    print()
    
    # 3. Monte Carlo Off-Policy Control (Unweighted)
    Q_mc_off_unw, steps_mc_off_unw = monte_carlo_off_policy_unweighted(
        grid_env, total_episodes=num_episodes, discount_factor=discount_factor,
        epsilon_val=epsilon_val, learning_rate=learning_rate
    )
    results_storage['MC Off-Policy (Unweighted)'] = steps_mc_off_unw
    Q_functions['MC Off-Policy (Unweighted)'] = Q_mc_off_unw
    print()
    
    # 4. Monte Carlo Off-Policy Control (Weighted)
    Q_mc_off_w, steps_mc_off_w = monte_carlo_off_policy_weighted(
        grid_env, total_episodes=num_episodes, discount_factor=discount_factor,
        epsilon_val=epsilon_val, learning_rate=learning_rate
    )
    results_storage['MC Off-Policy (Weighted)'] = steps_mc_off_w
    Q_functions['MC Off-Policy (Weighted)'] = Q_mc_off_w
    print()
    
    # 5. TD(0) On-Policy Control (SARSA)
    Q_td_on, steps_td_on = td_sarsa_control(
        grid_env, total_episodes=num_episodes, discount_factor=discount_factor,
        epsilon_val=epsilon_val, learning_rate=learning_rate
    )
    results_storage['TD(0) On-Policy (SARSA)'] = steps_td_on
    Q_functions['TD(0) On-Policy'] = Q_td_on
    print()
    
    # 6. TD(0) Off-Policy Control (Q-Learning)
    Q_td_off, steps_td_off = td_q_learning_control(
        grid_env, total_episodes=num_episodes, discount_factor=discount_factor,
        epsilon_val=epsilon_val, learning_rate=learning_rate
    )
    results_storage['TD(0) Off-Policy (Q-Learning)'] = steps_td_off
    Q_functions['TD(0) Off-Policy'] = Q_td_off
    print()
    
    # Generate performance comparison plot
    print("=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    plot_algorithm_comparison(results_storage, "Windy Gridworld: Control Algorithm Comparison")
    
    # Display sample learned policy
    print("\n" + "=" * 70)
    print("LEARNED POLICY EXAMPLE (SARSA)")
    print("=" * 70)
    display_learned_strategy(grid_env, Q_td_on)
    
    # Final performance summary
    print("=" * 70)
    print("FINAL PERFORMANCE SUMMARY (Last 100 Episodes)")
    print("=" * 70)
    for algorithm, steps in results_storage.items():
        final_perf = np.mean(steps[-100:])
        print(f"  {algorithm:30s}: {final_perf:6.2f} steps")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)
    
    return results_storage, Q_functions

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Run the complete experiment
    results, Q_dicts = run_comprehensive_experiment()