import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Tuple, List, Dict
import copy
 
# ============================================================================
# ENVIRONMENT: WINDY GRIDWORLD
# ============================================================================
 
class WindyGridworld:
    """
    Stochastic Windy Gridworld Environment
    - N x N grid with terminal state
    - Wind pushes agent upward (stochastically)
    - Reward: -1 for all transitions, 0 for reaching goal
    """
    
    def __init__(self, height=7, width=10, start=(3, 0), goal=(3, 7),
                 wind=None, walls=None):
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        
        # Default wind pattern (stronger in middle columns)
        if wind is None:
            self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        else:
            self.wind = wind
            
        self.walls = walls if walls is not None else set()
        
        # Actions: up, down, left, right
        self.actions = [0, 1, 2, 3]
        self.action_deltas = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        self.current_state = self.start
        
    def reset(self):
        """Reset environment to start state"""
        self.current_state = self.start
        return self.current_state
    
    def is_terminal(self, state):
        """Check if state is terminal (goal)"""
        return state == self.goal
    
    def get_stochastic_wind(self, col):
        """
        Stochastic wind for column:
        - 10% chance: wind + 1
        - 80% chance: normal wind
        - 10% chance: no wind
        """
        rand = np.random.random()
        if rand < 0.1:
            return self.wind[col] + 1
        elif rand < 0.9:
            return self.wind[col]
        else:
            return 0
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done)
        """
        if self.is_terminal(self.current_state):
            return self.current_state, 0, True
        
        r, c = self.current_state
        
        # Step 1: Apply action
        dr, dc = self.action_deltas[action]
        r_tilde, c_tilde = r + dr, c + dc
        
        # Step 2: Apply stochastic wind
        wind_strength = self.get_stochastic_wind(c)
        r_hat = r_tilde - wind_strength
        c_hat = c_tilde
        
        # Step 3: Clip to bounds
        r_prime = np.clip(r_hat, 0, self.height - 1)
        c_prime = np.clip(c_hat, 0, self.width - 1)
        
        # Step 4: Check walls
        if (r_prime, c_prime) in self.walls:
            r_prime, c_prime = r, c
        
        next_state = (r_prime, c_prime)
        
        # Reward: 0 if reaching goal, -1 otherwise
        reward = 0 if next_state == self.goal else -1
        done = self.is_terminal(next_state)
        
        self.current_state = next_state
        return next_state, reward, done
    
    def get_all_states(self):
        """Get all non-terminal states"""
        states = []
        for r in range(self.height):
            for c in range(self.width):
                if (r, c) != self.goal and (r, c) not in self.walls:
                    states.append((r, c))
        return states
 
 
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
 
def epsilon_greedy_policy(Q, state, actions, epsilon):
    """
    Epsilon-greedy policy
    """
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    else:
        q_values = [Q.get((state, a), 0.0) for a in actions]
        max_q = max(q_values)
        # Handle ties randomly
        best_actions = [a for a in actions if Q.get((state, a), 0.0) == max_q]
        return np.random.choice(best_actions)
 
def greedy_policy(Q, state, actions):
    """
    Greedy policy (deterministic)
    """
    q_values = [Q.get((state, a), 0.0) for a in actions]
    max_q = max(q_values)
    best_actions = [a for a in actions if Q.get((state, a), 0.0) == max_q]
    return np.random.choice(best_actions)
 
def generate_episode(env, policy, Q=None, epsilon=0.1):
    """
    Generate an episode following the given policy
    Returns: list of (state, action, reward) tuples
    """
    episode = []
    state = env.reset()
    
    while True:
        if Q is not None:
            action = policy(Q, state, env.actions, epsilon)
        else:
            action = policy(state, env.actions)
        
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        
        if done:
            break
        state = next_state
    
    return episode
 
 
# ============================================================================
# ALGORITHM 1: DYNAMIC PROGRAMMING CONTROL (Value Iteration)
# ============================================================================
 
def dp_control(env, gamma=1.0, theta=1e-6, max_iterations=1000):
    """
    Dynamic Programming Control using Value Iteration
    Requires model knowledge (transition probabilities)
    """
    print("Running DP Control...")
    
    # Initialize value function
    V = defaultdict(float)
    policy = {}
    
    # Get all states
    all_states = env.get_all_states()
    all_states.append(env.goal)  # Include goal state
    
    for iteration in range(max_iterations):
        delta = 0
        
        # Update value for each state
        for state in all_states:
            if env.is_terminal(state):
                V[state] = 0
                continue
            
            # Compute Q-values for all actions
            q_values = []
            for action in env.actions:
                # Since environment is stochastic, we average over wind outcomes
                q_sa = 0
                wind_probs = [0.1, 0.8, 0.1]  # [wind+1, normal, no wind]
                wind_values = [1, 0, -1]  # relative to base wind
                
                r, c = state
                dr, dc = env.action_deltas[action]
                
                for wind_prob, wind_delta in zip(wind_probs, wind_values):
                    # Calculate next state with this wind outcome
                    if wind_delta == 1:
                        wind_strength = env.wind[c] + 1
                    elif wind_delta == 0:
                        wind_strength = env.wind[c]
                    else:
                        wind_strength = 0
                    
                    r_next = np.clip(r + dr - wind_strength, 0, env.height - 1)
                    c_next = np.clip(c + dc, 0, env.width - 1)
                    
                    if (r_next, c_next) in env.walls:
                        r_next, c_next = r, c
                    
                    next_state = (r_next, c_next)
                    reward = 0 if next_state == env.goal else -1
                    
                    q_sa += wind_prob * (reward + gamma * V[next_state])
                
                q_values.append(q_sa)
            
            v_new = max(q_values)
            delta = max(delta, abs(V[state] - v_new))
            V[state] = v_new
        
        if delta < theta:
            print(f"  DP converged in {iteration + 1} iterations")
            break
    
    # Extract policy from value function
    Q = defaultdict(float)
    for state in all_states:
        if env.is_terminal(state):
            continue
        
        for action in env.actions:
            q_sa = 0
            wind_probs = [0.1, 0.8, 0.1]
            wind_values = [1, 0, -1]
            
            r, c = state
            dr, dc = env.action_deltas[action]
            
            for wind_prob, wind_delta in zip(wind_probs, wind_values):
                if wind_delta == 1:
                    wind_strength = env.wind[c] + 1
                elif wind_delta == 0:
                    wind_strength = env.wind[c]
                else:
                    wind_strength = 0
                
                r_next = np.clip(r + dr - wind_strength, 0, env.height - 1)
                c_next = np.clip(c + dc, 0, env.width - 1)
                
                if (r_next, c_next) in env.walls:
                    r_next, c_next = r, c
                
                next_state = (r_next, c_next)
                reward = 0 if next_state == env.goal else -1
                
                q_sa += wind_prob * (reward + gamma * V[next_state])
            
            Q[(state, action)] = q_sa
    
    return Q, V
 
 
# ============================================================================
# ALGORITHM 2: MONTE CARLO ON-POLICY CONTROL
# ============================================================================
 
def mc_on_policy_control(env, num_episodes=5000, gamma=1.0, epsilon=0.1, alpha=0.1):
    """
    Monte Carlo On-Policy Control (epsilon-greedy)
    """
    print("Running MC On-Policy Control...")
    
    Q = defaultdict(float)
    returns_count = defaultdict(int)
    episode_lengths = []
    
    for episode_num in range(num_episodes):
        # Generate episode using epsilon-greedy policy
        episode = generate_episode(env, epsilon_greedy_policy, Q, epsilon)
        episode_lengths.append(len(episode))
        
        # Track states-actions visited in this episode
        visited = set()
        
        # Calculate returns and update Q (first-visit)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            if (state, action) not in visited:
                visited.add((state, action))
                # Incremental update
                returns_count[(state, action)] += 1
                Q[(state, action)] += alpha * (G - Q[(state, action)])
        
        if (episode_num + 1) % 500 == 0:
            avg_length = np.mean(episode_lengths[-500:])
            print(f"  Episode {episode_num + 1}/{num_episodes}, Avg Length: {avg_length:.2f}")
    
    return Q, episode_lengths
 
 
# ============================================================================
# ALGORITHM 3: MONTE CARLO OFF-POLICY CONTROL
# ============================================================================
 
def mc_off_policy_control(env, num_episodes=5000, gamma=1.0, epsilon=0.1, alpha=0.1):
    """
    Monte Carlo Off-Policy Control with Importance Sampling
    Behavior policy: epsilon-greedy
    Target policy: greedy
    """
    print("Running MC Off-Policy Control...")
    
    Q = defaultdict(float)
    C = defaultdict(float)  # Cumulative sum of weights
    episode_lengths = []
    
    for episode_num in range(num_episodes):
        # Generate episode using behavior policy (epsilon-greedy)
        episode = generate_episode(env, epsilon_greedy_policy, Q, epsilon)
        episode_lengths.append(len(episode))
        
        G = 0
        W = 1.0  # Importance sampling ratio
        
        # Update in reverse
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            C[(state, action)] += W
            Q[(state, action)] += (W / C[(state, action)]) * (G - Q[(state, action)])
            
            # Target policy is greedy
            greedy_action = greedy_policy(Q, state, env.actions)
            if action != greedy_action:
                break  # Importance sampling ratio becomes 0
            
            # Update importance sampling ratio
            # P(target) / P(behavior)
            # Target policy: greedy (probability 1 for greedy action)
            # Behavior policy: epsilon-greedy
            prob_behavior = epsilon / len(env.actions) + (1 - epsilon) if action == greedy_action else epsilon / len(env.actions)
            prob_target = 1.0
            W *= prob_target / prob_behavior
        
        if (episode_num + 1) % 500 == 0:
            avg_length = np.mean(episode_lengths[-500:])
            print(f"  Episode {episode_num + 1}/{num_episodes}, Avg Length: {avg_length:.2f}")
    
    return Q, episode_lengths
 
 
# ============================================================================
# ALGORITHM 4: TD(0) ON-POLICY CONTROL (SARSA)
# ============================================================================
 
def td_on_policy_control(env, num_episodes=5000, gamma=1.0, epsilon=0.1, alpha=0.1):
    """
    TD(0) On-Policy Control (SARSA)
    """
    print("Running TD(0) On-Policy Control (SARSA)...")
    
    Q = defaultdict(float)
    episode_lengths = []
    
    for episode_num in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, env.actions, epsilon)
        
        steps = 0
        while True:
            next_state, reward, done = env.step(action)
            steps += 1
            
            if done:
                # Update for terminal state
                Q[(state, action)] += alpha * (reward - Q[(state, action)])
                break
            
            # Choose next action
            next_action = epsilon_greedy_policy(Q, next_state, env.actions, epsilon)
            
            # SARSA update
            td_target = reward + gamma * Q[(next_state, next_action)]
            Q[(state, action)] += alpha * (td_target - Q[(state, action)])
            
            state = next_state
            action = next_action
        
        episode_lengths.append(steps)
        
        if (episode_num + 1) % 500 == 0:
            avg_length = np.mean(episode_lengths[-500:])
            print(f"  Episode {episode_num + 1}/{num_episodes}, Avg Length: {avg_length:.2f}")
    
    return Q, episode_lengths
 
 
# ============================================================================
# ALGORITHM 5: TD(0) OFF-POLICY CONTROL (Q-Learning, Unweighted IS)
# ============================================================================
 
def td_off_policy_unweighted(env, num_episodes=5000, gamma=1.0, epsilon=0.1, alpha=0.1):
    """
    TD(0) Off-Policy Control with Unweighted Importance Sampling
    This is essentially Q-Learning
    """
    print("Running TD(0) Off-Policy Control (Unweighted IS / Q-Learning)...")
    
    Q = defaultdict(float)
    episode_lengths = []
    
    for episode_num in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            # Behavior policy: epsilon-greedy
            action = epsilon_greedy_policy(Q, state, env.actions, epsilon)
            next_state, reward, done = env.step(action)
            steps += 1
            
            if done:
                Q[(state, action)] += alpha * (reward - Q[(state, action)])
                break
            
            # Target policy: greedy (max over next actions)
            max_q_next = max([Q.get((next_state, a), 0.0) for a in env.actions])
            
            # Q-Learning update
            td_target = reward + gamma * max_q_next
            Q[(state, action)] += alpha * (td_target - Q[(state, action)])
            
            state = next_state
        
        episode_lengths.append(steps)
        
        if (episode_num + 1) % 500 == 0:
            avg_length = np.mean(episode_lengths[-500:])
            print(f"  Episode {episode_num + 1}/{num_episodes}, Avg Length: {avg_length:.2f}")
    
    return Q, episode_lengths
 
 
# ============================================================================
# ALGORITHM 6: TD(0) OFF-POLICY CONTROL (Weighted IS)
# ============================================================================
 
def td_off_policy_weighted(env, num_episodes=5000, gamma=1.0, epsilon=0.1, alpha=0.1):
    """
    TD(0) Off-Policy Control with Weighted Importance Sampling
    """
    print("Running TD(0) Off-Policy Control (Weighted IS)...")
    
    Q = defaultdict(float)
    C = defaultdict(float)  # Cumulative weights
    episode_lengths = []
    
    for episode_num in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            # Behavior policy: epsilon-greedy
            action = epsilon_greedy_policy(Q, state, env.actions, epsilon)
            next_state, reward, done = env.step(action)
            steps += 1
            
            # Calculate importance sampling ratio
            greedy_action = greedy_policy(Q, state, env.actions)
            if action == greedy_action:
                prob_behavior = epsilon / len(env.actions) + (1 - epsilon)
            else:
                prob_behavior = epsilon / len(env.actions)
            
            prob_target = 1.0 if action == greedy_action else 0.0
            
            if prob_target == 0:
                # Can't learn from this transition
                if done:
                    break
                state = next_state
                continue
            
            rho = prob_target / prob_behavior
            
            if done:
                C[(state, action)] += rho
                Q[(state, action)] += (rho / C[(state, action)]) * (reward - Q[(state, action)])
                break
            
            # Target policy: greedy
            max_q_next = max([Q.get((next_state, a), 0.0) for a in env.actions])
            
            # Weighted IS update
            td_target = reward + gamma * max_q_next
            C[(state, action)] += rho
            Q[(state, action)] += (rho / C[(state, action)]) * (td_target - Q[(state, action)])
            
            state = next_state
        
        episode_lengths.append(steps)
        
        if (episode_num + 1) % 500 == 0:
            avg_length = np.mean(episode_lengths[-500:])
            print(f"  Episode {episode_num + 1}/{num_episodes}, Avg Length: {avg_length:.2f}")
    
    return Q, episode_lengths
 
 
# ============================================================================
# VISUALIZATION AND COMPARISON
# ============================================================================
 
def plot_results(results_dict, title="Algorithm Comparison"):
    """
    Plot learning curves for all algorithms
    """
    plt.figure(figsize=(12, 6))
    
    for name, episode_lengths in results_dict.items():
        # Smooth the curve with moving average
        window = 100
        smoothed = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=name, linewidth=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Steps per Episode', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
 
def visualize_policy(env, Q):
    """
    Visualize the learned policy as arrows on the grid
    """
    action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    grid = [['' for _ in range(env.width)] for _ in range(env.height)]
    
    for r in range(env.height):
        for c in range(env.width):
            state = (r, c)
            if state == env.goal:
                grid[r][c] = 'G'
            elif state == env.start:
                q_values = [Q.get((state, a), 0.0) for a in env.actions]
                best_action = env.actions[np.argmax(q_values)]
                grid[r][c] = 'S' + action_symbols[best_action]
            else:
                q_values = [Q.get((state, a), 0.0) for a in env.actions]
                best_action = env.actions[np.argmax(q_values)]
                grid[r][c] = action_symbols[best_action]
    
    print("\nLearned Policy (arrows show best action):")
    print("S = Start, G = Goal")
    for r in range(env.height):
        print(' '.join(f'{grid[r][c]:>2}' for c in range(env.width)))
    print()
 
 
# ============================================================================
# MAIN EXECUTION
# ============================================================================
 
def main():
    """
    Main function to run all algorithms and compare results
    """
    print("="*70)
    print("WINDY GRIDWORLD - REINFORCEMENT LEARNING ASSIGNMENT")
    print("="*70)
    
    # Create environment
    env = WindyGridworld(
        height=7,
        width=10,
        start=(3, 0),
        goal=(3, 7),
        wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    )
    
    print(f"\nEnvironment Setup:")
    print(f"  Grid Size: {env.height} x {env.width}")
    print(f"  Start: {env.start}")
    print(f"  Goal: {env.goal}")
    print(f"  Wind Pattern: {env.wind}")
    print()
    
    # Hyperparameters
    num_episodes = 3000
    gamma = 1.0
    epsilon = 0.1
    alpha = 0.1
    
    results = {}
    learned_Q = {}
    
    # Run all algorithms
    print("\n" + "="*70)
    print("RUNNING ALL ALGORITHMS")
    print("="*70 + "\n")
    
    # 1. DP Control
    Q_dp, V_dp = dp_control(env, gamma=gamma)
    learned_Q['DP'] = Q_dp
    print()
    
    # 2. MC On-Policy
    Q_mc_on, lengths_mc_on = mc_on_policy_control(
        env, num_episodes=num_episodes, gamma=gamma, epsilon=epsilon, alpha=alpha
    )
    results['MC On-Policy'] = lengths_mc_on
    learned_Q['MC On-Policy'] = Q_mc_on
    print()
    
    # 3. MC Off-Policy
    Q_mc_off, lengths_mc_off = mc_off_policy_control(
        env, num_episodes=num_episodes, gamma=gamma, epsilon=epsilon, alpha=alpha
    )
    results['MC Off-Policy'] = lengths_mc_off
    learned_Q['MC Off-Policy'] = Q_mc_off
    print()
    
    # 4. TD On-Policy (SARSA)
    Q_td_on, lengths_td_on = td_on_policy_control(
        env, num_episodes=num_episodes, gamma=gamma, epsilon=epsilon, alpha=alpha
    )
    results['TD(0) On-Policy (SARSA)'] = lengths_td_on
    learned_Q['TD(0) On-Policy'] = Q_td_on
    print()
    
    # 5. TD Off-Policy Unweighted (Q-Learning)
    Q_td_off_unw, lengths_td_off_unw = td_off_policy_unweighted(
        env, num_episodes=num_episodes, gamma=gamma, epsilon=epsilon, alpha=alpha
    )
    results['TD(0) Off-Policy (Unweighted)'] = lengths_td_off_unw
    learned_Q['TD(0) Off-Policy Unweighted'] = Q_td_off_unw
    print()
    
    # 6. TD Off-Policy Weighted
    Q_td_off_w, lengths_td_off_w = td_off_policy_weighted(
        env, num_episodes=num_episodes, gamma=gamma, epsilon=epsilon, alpha=alpha
    )
    results['TD(0) Off-Policy (Weighted)'] = lengths_td_off_w
    learned_Q['TD(0) Off-Policy Weighted'] = Q_td_off_w
    print()
    
    # Plot comparison
    print("="*70)
    print("RESULTS")
    print("="*70)
    plot_results(results, "Windy Gridworld: Algorithm Comparison")
    
    # Visualize one of the learned policies
    print("\n" + "="*70)
    print("SAMPLE LEARNED POLICY (TD(0) On-Policy)")
    print("="*70)
    visualize_policy(env, Q_td_on)
    
    # Print final statistics
    print("="*70)
    print("FINAL PERFORMANCE (Last 100 Episodes Average)")
    print("="*70)
    for name, lengths in results.items():
        avg_last_100 = np.mean(lengths[-100:])
        print(f"  {name:35s}: {avg_last_100:6.2f} steps")
    print()
    
    print("="*70)
    print("ASSIGNMENT COMPLETE!")
    print("="*70)
    
    return results, learned_Q
 
 
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run main experiment
    results, learned_Q = main()