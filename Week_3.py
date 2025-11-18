import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set
import random

# ============================================================================
# ENVIRONMENT: GUSTY GRID NAVIGATION
# ============================================================================

class GustyGridNavigation:
    """
    Grid Navigation with Gusty Winds Environment
    - Grid with goal state and variable wind effects
    - Stochastic wind movements
    - Penalty for each step, reward for reaching destination
    """
    
    def __init__(self, rows=7, cols=10, initial_pos=(3, 0), target=(3, 7),
                 gusts=None, obstacles=None):
        self.rows = rows
        self.cols = cols
        self.initial_pos = initial_pos
        self.target = target
        
        # Wind gust pattern (different representation)
        if gusts is None:
            self.gusts = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:1, 9:0}
        else:
            self.gusts = gusts
            
        self.obstacles = obstacles if obstacles is not None else set()
        
        # Movement directions: north, south, west, east
        self.moves = ['N', 'S', 'W', 'E']
        self.move_offsets = {
            'N': (-1, 0),
            'S': (1, 0),
            'W': (0, -1),
            'E': (0, 1)
        }
        
        self.current_position = self.initial_pos
    
    def restart(self):
        """Reset to starting position"""
        self.current_position = self.initial_pos
        return self.current_position
    
    def reached_target(self, position):
        """Check if position is target"""
        return position == self.target
    
    def get_wind_effect(self, column):
        """
        Calculate wind effect for column with randomness
        - 10% probability: extra strong gust
        - 80% probability: normal gust
        - 10% probability: calm (no gust)
        """
        chance = random.uniform(0, 1)
        base_gust = self.gusts.get(column, 0)
        
        if chance < 0.1:
            return base_gust + 1
        elif chance < 0.9:
            return base_gust
        else:
            return 0
    
    def make_move(self, direction):
        """
        Execute movement and return (new_position, score, completed)
        """
        if self.reached_target(self.current_position):
            return self.current_position, 0, True
        
        row, col = self.current_position
        
        # Apply movement
        dr, dc = self.move_offsets[direction]
        new_row, new_col = row + dr, col + dc
        
        # Apply wind effect
        gust_power = self.get_wind_effect(col)
        wind_affected_row = new_row - gust_power
        wind_affected_col = new_col
        
        # Ensure within bounds
        final_row = max(0, min(wind_affected_row, self.rows - 1))
        final_col = max(0, min(wind_affected_col, self.cols - 1))
        
        # Check for obstacles
        if (final_row, final_col) in self.obstacles:
            final_row, final_col = row, col
        
        new_position = (final_row, final_col)
        
        # Calculate reward
        score = 0 if new_position == self.target else -1
        completed = self.reached_target(new_position)
        
        self.current_position = new_position
        return new_position, score, completed
    
    def get_valid_positions(self):
        """Get all non-target positions"""
        positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) != self.target and (r, c) not in self.obstacles:
                    positions.append((r, c))
        return positions

# ============================================================================
# POLICY AND EPISODE FUNCTIONS
# ============================================================================

def select_action_eps_greedy(Q_values, position, available_actions, eps):
    """
    Select action using epsilon-greedy strategy
    """
    if random.random() < eps:
        return random.choice(available_actions)
    else:
        # Get Q-values for current position
        action_values = [Q_values.get((position, act), 0.0) for act in available_actions]
        best_value = max(action_values)
        # Handle multiple best actions
        optimal_actions = [act for act in available_actions 
                          if Q_values.get((position, act), 0.0) == best_value]
        return random.choice(optimal_actions)

def select_action_greedy(Q_values, position, available_actions):
    """
    Select action greedily
    """
    action_values = [Q_values.get((position, act), 0.0) for act in available_actions]
    best_value = max(action_values)
    optimal_actions = [act for act in available_actions 
                      if Q_values.get((position, act), 0.0) == best_value]
    return random.choice(optimal_actions)

def run_episode(environment, policy_func, Q_dict=None, eps=0.1):
    """
    Run one episode following policy
    Returns: sequence of (position, action, reward)
    """
    trajectory = []
    current_pos = environment.restart()
    
    while True:
        if Q_dict is not None:
            action_taken = policy_func(Q_dict, current_pos, environment.moves, eps)
        else:
            action_taken = policy_func(current_pos, environment.moves)
        
        next_pos, reward, done = environment.make_move(action_taken)
        trajectory.append((current_pos, action_taken, reward))
        
        if done:
            break
        current_pos = next_pos
    
    return trajectory

# ============================================================================
# ALGORITHM 1: VALUE ITERATION
# ============================================================================

def run_value_iteration(environment, discount=1.0, tolerance=1e-6, max_iters=1000):
    """
    Dynamic Programming using Value Iteration
    Requires environment model knowledge
    """
    print("Executing Value Iteration...")
    
    # Initialize state values
    V_vals = {}
    policy_map = {}
    
    # Get all states
    states_list = environment.get_valid_positions()
    states_list.append(environment.target)
    
    for iteration in range(max_iters):
        max_change = 0
        
        for state in states_list:
            if environment.reached_target(state):
                V_vals[state] = 0
                continue
            
            # Calculate Q-values for each action
            q_vals_per_action = []
            for action in environment.moves:
                q_val = 0
                # Wind outcome probabilities
                prob_dist = [0.1, 0.8, 0.1]
                wind_modifiers = [1, 0, -1]
                
                r, c = state
                dr, dc = environment.move_offsets[action]
                
                for prob, wind_mod in zip(prob_dist, wind_modifiers):
                    # Determine wind strength
                    if wind_mod == 1:
                        gust_strength = environment.gusts[c] + 1
                    elif wind_mod == 0:
                        gust_strength = environment.gusts[c]
                    else:
                        gust_strength = 0
                    
                    # Calculate resulting position
                    r_result = r + dr - gust_strength
                    c_result = c + dc
                    
                    # Apply bounds and obstacles
                    r_final = max(0, min(r_result, environment.rows - 1))
                    c_final = max(0, min(c_result, environment.cols - 1))
                    
                    if (r_final, c_final) in environment.obstacles:
                        r_final, c_final = r, c
                    
                    result_state = (r_final, c_final)
                    reward_val = 0 if result_state == environment.target else -1
                    
                    q_val += prob * (reward_val + discount * V_vals.get(result_state, 0.0))
                
                q_vals_per_action.append(q_val)
            
            new_v = max(q_vals_per_action)
            max_change = max(max_change, abs(V_vals.get(state, 0.0) - new_v))
            V_vals[state] = new_v
        
        if max_change < tolerance:
            print(f"  Value Iteration completed in {iteration + 1} iterations")
            break
    
    # Extract Q-values from value function
    Q_dict = {}
    for state in states_list:
        if environment.reached_target(state):
            continue
        
        for action in environment.moves:
            q_val = 0
            prob_dist = [0.1, 0.8, 0.1]
            wind_modifiers = [1, 0, -1]
            
            r, c = state
            dr, dc = environment.move_offsets[action]
            
            for prob, wind_mod in zip(prob_dist, wind_modifiers):
                if wind_mod == 1:
                    gust_strength = environment.gusts[c] + 1
                elif wind_mod == 0:
                    gust_strength = environment.gusts[c]
                else:
                    gust_strength = 0
                
                r_result = r + dr - gust_strength
                c_result = c + dc
                
                r_final = max(0, min(r_result, environment.rows - 1))
                c_final = max(0, min(c_result, environment.cols - 1))
                
                if (r_final, c_final) in environment.obstacles:
                    r_final, c_final = r, c
                
                result_state = (r_final, c_final)
                reward_val = 0 if result_state == environment.target else -1
                
                q_val += prob * (reward_val + discount * V_vals.get(result_state, 0.0))
            
            Q_dict[(state, action)] = q_val
    
    return Q_dict, V_vals

# ============================================================================
# ALGORITHM 2: MONTE CARLO ON-POLICY
# ============================================================================

def monte_carlo_on_policy(environment, episodes=5000, discount=1.0, eps=0.1, step_size=0.1):
    """
    Monte Carlo On-Policy Control with epsilon-greedy policy
    """
    print("Running Monte Carlo On-Policy...")
    
    Q_dict = {}
    visit_counts = {}
    episode_steps = []
    
    for episode_idx in range(episodes):
        # Generate episode
        episode_data = run_episode(environment, select_action_eps_greedy, Q_dict, eps)
        episode_steps.append(len(episode_data))
        
        # Track visited state-action pairs
        visited_pairs = set()
        
        # Calculate returns and update Q-values
        cumulative_return = 0
        for step in reversed(range(len(episode_data))):
            state, action, reward = episode_data[step]
            cumulative_return = discount * cumulative_return + reward
            
            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                # Update visit count and Q-value
                visit_counts[(state, action)] = visit_counts.get((state, action), 0) + 1
                current_q = Q_dict.get((state, action), 0.0)
                Q_dict[(state, action)] = current_q + step_size * (cumulative_return - current_q)
        
        if (episode_idx + 1) % 500 == 0:
            avg_steps = np.mean(episode_steps[-500:])
            print(f"  Episode {episode_idx + 1}/{episodes}, Average Steps: {avg_steps:.2f}")
    
    return Q_dict, episode_steps

# ============================================================================
# ALGORITHM 3: MONTE CARLO OFF-POLICY
# ============================================================================

def monte_carlo_off_policy(environment, episodes=5000, discount=1.0, eps=0.1, step_size=0.1):
    """
    Monte Carlo Off-Policy Control with Importance Sampling
    """
    print("Running Monte Carlo Off-Policy...")
    
    Q_dict = {}
    weight_sums = {}
    episode_steps = []
    
    for episode_idx in range(episodes):
        # Generate episode with behavior policy
        episode_data = run_episode(environment, select_action_eps_greedy, Q_dict, eps)
        episode_steps.append(len(episode_data))
        
        cumulative_return = 0
        importance_ratio = 1.0
        
        # Process episode backwards
        for step in reversed(range(len(episode_data))):
            state, action, reward = episode_data[step]
            cumulative_return = discount * cumulative_return + reward
            
            # Update Q-value with importance sampling
            weight_sums[(state, action)] = weight_sums.get((state, action), 0.0) + importance_ratio
            current_q = Q_dict.get((state, action), 0.0)
            Q_dict[(state, action)] = current_q + (importance_ratio / weight_sums[(state, action)]) * (cumulative_return - current_q)
            
            # Check if action matches target policy
            target_action = select_action_greedy(Q_dict, state, environment.moves)
            if action != target_action:
                break  # Stop if behavior diverges from target
            
            # Update importance sampling ratio
            # Target policy probability
            target_prob = 1.0
            # Behavior policy probability
            if action == target_action:
                behavior_prob = eps / len(environment.moves) + (1 - eps)
            else:
                behavior_prob = eps / len(environment.moves)
            
            importance_ratio *= target_prob / behavior_prob
        
        if (episode_idx + 1) % 500 == 0:
            avg_steps = np.mean(episode_steps[-500:])
            print(f"  Episode {episode_idx + 1}/{episodes}, Average Steps: {avg_steps:.2f}")
    
    return Q_dict, episode_steps

# ============================================================================
# ALGORITHM 4: SARSA (ON-POLICY TD)
# ============================================================================

def sarsa_learning(environment, episodes=5000, discount=1.0, eps=0.1, step_size=0.1):
    """
    SARSA: On-Policy TD Control
    """
    print("Running SARSA Learning...")
    
    Q_dict = {}
    episode_steps = []
    
    for episode_idx in range(episodes):
        current_state = environment.restart()
        current_action = select_action_eps_greedy(Q_dict, current_state, environment.moves, eps)
        
        steps = 0
        while True:
            next_state, reward, done = environment.make_move(current_action)
            steps += 1
            
            if done:
                # Update Q-value for terminal transition
                Q_dict[(current_state, current_action)] = Q_dict.get((current_state, current_action), 0.0) + \
                    step_size * (reward - Q_dict.get((current_state, current_action), 0.0))
                break
            
            # Choose next action
            next_action = select_action_eps_greedy(Q_dict, next_state, environment.moves, eps)
            
            # SARSA update
            td_target = reward + discount * Q_dict.get((next_state, next_action), 0.0)
            td_error = td_target - Q_dict.get((current_state, current_action), 0.0)
            Q_dict[(current_state, current_action)] = Q_dict.get((current_state, current_action), 0.0) + step_size * td_error
            
            current_state = next_state
            current_action = next_action
        
        episode_steps.append(steps)
        
        if (episode_idx + 1) % 500 == 0:
            avg_steps = np.mean(episode_steps[-500:])
            print(f"  Episode {episode_idx + 1}/{episodes}, Average Steps: {avg_steps:.2f}")
    
    return Q_dict, episode_steps

# ============================================================================
# ALGORITHM 5: Q-LEARNING (OFF-POLICY TD)
# ============================================================================

def q_learning(environment, episodes=5000, discount=1.0, eps=0.1, step_size=0.1):
    """
    Q-Learning: Off-Policy TD Control
    """
    print("Running Q-Learning...")
    
    Q_dict = {}
    episode_steps = []
    
    for episode_idx in range(episodes):
        current_state = environment.restart()
        
        steps = 0
        while True:
            # Behavior policy: epsilon-greedy
            current_action = select_action_eps_greedy(Q_dict, current_state, environment.moves, eps)
            next_state, reward, done = environment.make_move(current_action)
            steps += 1
            
            if done:
                # Update for terminal state
                Q_dict[(current_state, current_action)] = Q_dict.get((current_state, current_action), 0.0) + \
                    step_size * (reward - Q_dict.get((current_state, current_action), 0.0))
                break
            
            # Target policy: greedy (max Q)
            max_next_q = max([Q_dict.get((next_state, act), 0.0) for act in environment.moves])
            
            # Q-Learning update
            td_target = reward + discount * max_next_q
            td_error = td_target - Q_dict.get((current_state, current_action), 0.0)
            Q_dict[(current_state, current_action)] = Q_dict.get((current_state, current_action), 0.0) + step_size * td_error
            
            current_state = next_state
        
        episode_steps.append(steps)
        
        if (episode_idx + 1) % 500 == 0:
            avg_steps = np.mean(episode_steps[-500:])
            print(f"  Episode {episode_idx + 1}/{episodes}, Average Steps: {avg_steps:.2f}")
    
    return Q_dict, episode_steps

# ============================================================================
# ALGORITHM 6: WEIGHTED IS TD LEARNING
# ============================================================================

def weighted_is_td_learning(environment, episodes=5000, discount=1.0, eps=0.1, step_size=0.1):
    """
    TD Learning with Weighted Importance Sampling
    """
    print("Running Weighted IS TD Learning...")
    
    Q_dict = {}
    cumulative_weights = {}
    episode_steps = []
    
    for episode_idx in range(episodes):
        current_state = environment.restart()
        
        steps = 0
        while True:
            # Behavior policy
            current_action = select_action_eps_greedy(Q_dict, current_state, environment.moves, eps)
            next_state, reward, done = environment.make_move(current_action)
            steps += 1
            
            # Calculate importance sampling ratio
            greedy_act = select_action_greedy(Q_dict, current_state, environment.moves)
            if current_action == greedy_act:
                behavior_prob = eps / len(environment.moves) + (1 - eps)
            else:
                behavior_prob = eps / len(environment.moves)
            
            target_prob = 1.0 if current_action == greedy_act else 0.0
            
            if target_prob == 0:
                # Skip update if action not in target policy
                if done:
                    break
                current_state = next_state
                continue
            
            is_ratio = target_prob / behavior_prob
            
            if done:
                cumulative_weights[(current_state, current_action)] = cumulative_weights.get((current_state, current_action), 0.0) + is_ratio
                old_q = Q_dict.get((current_state, current_action), 0.0)
                Q_dict[(current_state, current_action)] = old_q + \
                    (is_ratio / cumulative_weights[(current_state, current_action)]) * (reward - old_q)
                break
            
            # Target policy is greedy
            max_next_q = max([Q_dict.get((next_state, act), 0.0) for act in environment.moves])
            
            # Weighted update
            td_target = reward + discount * max_next_q
            cumulative_weights[(current_state, current_action)] = cumulative_weights.get((current_state, current_action), 0.0) + is_ratio
            old_q = Q_dict.get((current_state, current_action), 0.0)
            Q_dict[(current_state, current_action)] = old_q + \
                (is_ratio / cumulative_weights[(current_state, current_action)]) * (td_target - old_q)
            
            current_state = next_state
        
        episode_steps.append(steps)
        
        if (episode_idx + 1) % 500 == 0:
            avg_steps = np.mean(episode_steps[-500:])
            print(f"  Episode {episode_idx + 1}/{episodes}, Average Steps: {avg_steps:.2f}")
    
    return Q_dict, episode_steps

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def plot_learning_curves(results_data, title="Algorithm Performance Comparison"):
    """
    Plot learning curves for comparison
    """
    plt.figure(figsize=(12, 6))
    
    for algorithm_name, step_counts in results_data.items():
        # Apply smoothing
        window = 100
        if len(step_counts) >= window:
            smoothed = np.convolve(step_counts, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, label=algorithm_name, linewidth=2)
        else:
            plt.plot(step_counts, label=algorithm_name, linewidth=2)
    
    plt.xlabel('Episode Number', fontsize=12)
    plt.ylabel('Steps per Episode', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def display_policy(environment, Q_dict):
    """
    Display the learned policy visually
    """
    action_symbols = {'N': '↑', 'S': '↓', 'W': '←', 'E': '→'}
    
    policy_grid = [['' for _ in range(environment.cols)] for _ in range(environment.rows)]
    
    for r in range(environment.rows):
        for c in range(environment.cols):
            pos = (r, c)
            if pos == environment.target:
                policy_grid[r][c] = 'G'
            elif pos == environment.initial_pos:
                action_vals = [Q_dict.get((pos, act), 0.0) for act in environment.moves]
                best_act = environment.moves[np.argmax(action_vals)]
                policy_grid[r][c] = 'S' + action_symbols[best_act]
            else:
                action_vals = [Q_dict.get((pos, act), 0.0) for act in environment.moves]
                best_act = environment.moves[np.argmax(action_vals)]
                policy_grid[r][c] = action_symbols[best_act]
    
    print("\nLearned Policy Visualization:")
    print("S = Start, G = Goal")
    for row in policy_grid:
        print(' '.join(f'{cell:>2}' for cell in row))
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def execute_comparison():
    """
    Main function to run all algorithms and compare performance
    """
    print("="*70)
    print("GUSTY GRID NAVIGATION - RL ALGORITHMS COMPARISON")
    print("="*70)
    
    # Initialize environment
    grid_env = GustyGridNavigation(
        rows=7,
        cols=10,
        initial_pos=(3, 0),
        target=(3, 7),
        gusts={0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:1, 9:0}
    )
    
    print(f"\nEnvironment Configuration:")
    print(f"  Grid Dimensions: {grid_env.rows} x {grid_env.cols}")
    print(f"  Start Position: {grid_env.initial_pos}")
    print(f"  Target Position: {grid_env.target}")
    print(f"  Gust Pattern: {[grid_env.gusts[i] for i in range(grid_env.cols)]}")
    print()
    
    # Algorithm parameters
    total_episodes = 3000
    discount_factor = 1.0
    exploration_rate = 0.1
    learning_rate = 0.1
    
    results_data = {}
    Q_results = {}
    
    # Execute all algorithms
    print("\n" + "="*70)
    print("EXECUTING REINFORCEMENT LEARNING ALGORITHMS")
    print("="*70 + "\n")
    
    # 1. Value Iteration
    Q_vi, V_vi = run_value_iteration(grid_env, discount=discount_factor)
    Q_results['Value Iteration'] = Q_vi
    print()
    
    # 2. Monte Carlo On-Policy
    Q_mc_on, steps_mc_on = monte_carlo_on_policy(
        grid_env, episodes=total_episodes, discount=discount_factor, 
        eps=exploration_rate, step_size=learning_rate
    )
    results_data['Monte Carlo On-Policy'] = steps_mc_on
    Q_results['Monte Carlo On-Policy'] = Q_mc_on
    print()
    
    # 3. Monte Carlo Off-Policy
    Q_mc_off, steps_mc_off = monte_carlo_off_policy(
        grid_env, episodes=total_episodes, discount=discount_factor,
        eps=exploration_rate, step_size=learning_rate
    )
    results_data['Monte Carlo Off-Policy'] = steps_mc_off
    Q_results['Monte Carlo Off-Policy'] = Q_mc_off
    print()
    
    # 4. SARSA
    Q_sarsa, steps_sarsa = sarsa_learning(
        grid_env, episodes=total_episodes, discount=discount_factor,
        eps=exploration_rate, step_size=learning_rate
    )
    results_data['SARSA'] = steps_sarsa
    Q_results['SARSA'] = Q_sarsa
    print()
    
    # 5. Q-Learning
    Q_ql, steps_ql = q_learning(
        grid_env, episodes=total_episodes, discount=discount_factor,
        eps=exploration_rate, step_size=learning_rate
    )
    results_data['Q-Learning'] = steps_ql
    Q_results['Q-Learning'] = Q_ql
    print()
    
    # 6. Weighted IS TD
    Q_wis, steps_wis = weighted_is_td_learning(
        grid_env, episodes=total_episodes, discount=discount_factor,
        eps=exploration_rate, step_size=learning_rate
    )
    results_data['Weighted IS TD'] = steps_wis
    Q_results['Weighted IS TD'] = Q_wis
    print()
    
    # Generate comparison plot
    print("="*70)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*70)
    plot_learning_curves(results_data, "Gusty Grid Navigation: Algorithm Performance")
    
    # Show sample policy
    print("\n" + "="*70)
    print("SAMPLE LEARNED POLICY (SARSA)")
    print("="*70)
    display_policy(grid_env, Q_sarsa)
    
    # Final performance summary
    print("="*70)
    print("FINAL PERFORMANCE SUMMARY (Last 100 Episodes)")
    print("="*70)
    for algo_name, step_data in results_data.items():
        final_performance = np.mean(step_data[-100:])
        print(f"  {algo_name:30s}: {final_performance:6.2f} steps")
    print()
    
    print("="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    
    return results_data, Q_results

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run the comparison
    results, Q_values = execute_comparison()