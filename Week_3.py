import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import random
from dataclasses import dataclass
from enum import Enum


class ActionDirection(Enum):
    NORTH = 0
    SOUTH = 1  
    WEST = 2
    EAST = 3


@dataclass
class GridConfiguration:
    """Configuration parameters for grid environment"""
    rows: int = 7
    columns: int = 10
    start_position: Tuple[int, int] = (3, 0)
    goal_position: Tuple[int, int] = (3, 7)
    wind_pattern: List[int] = None
    reward_step: float = -1.0
    reward_goal: float = 0.0
    
    def __post_init__(self):
        if self.wind_pattern is None:
            self.wind_pattern = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]


class StochasticWindyGrid:
    """
    Advanced stochastic windy gridworld environment
    with configurable parameters and enhanced functionality
    """
    
    def __init__(self, config: GridConfiguration):
        self.config = config
        self.actions = list(ActionDirection)
        self.action_vectors = {
            ActionDirection.NORTH: (-1, 0),
            ActionDirection.SOUTH: (1, 0),
            ActionDirection.WEST: (0, -1),
            ActionDirection.EAST: (0, 1)
        }
        self.current_position = config.start_position
        
    def reset_agent(self) -> Tuple[int, int]:
        """Reset agent to starting position"""
        self.current_position = self.config.start_position
        return self.current_position
    
    def is_goal_state(self, state: Tuple[int, int]) -> bool:
        """Check if state is terminal goal state"""
        return state == self.config.goal_position
    
    def _get_wind_intensity(self, column: int) -> int:
        """
        Calculate stochastic wind intensity for given column
        Returns: wind strength with probabilistic variations
        """
        probability = random.random()
        base_wind = self.config.wind_pattern[column]
        
        if probability < 0.1:
            return base_wind + 1
        elif probability < 0.9:
            return base_wind
        else:
            return 0
    
    def _enforce_boundaries(self, row: int, col: int) -> Tuple[int, int]:
        """Ensure position stays within grid boundaries"""
        bounded_row = max(0, min(row, self.config.rows - 1))
        bounded_col = max(0, min(col, self.config.columns - 1))
        return bounded_row, bounded_col
    
    def execute_action(self, action: ActionDirection) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute action in current environment state
        Returns: (next_state, reward, episode_complete)
        """
        if self.is_goal_state(self.current_position):
            return self.current_position, self.config.reward_goal, True
        
        current_row, current_col = self.current_position
        row_delta, col_delta = self.action_vectors[action]
        
        # Phase 1: Apply movement action
        new_row = current_row + row_delta
        new_col = current_col + col_delta
        
        # Phase 2: Apply stochastic wind effect
        wind_force = self._get_wind_intensity(current_col)
        wind_affected_row = new_row - wind_force
        
        # Phase 3: Apply boundary constraints
        final_row, final_col = self._enforce_boundaries(wind_affected_row, new_col)
        resulting_state = (final_row, final_col)
        
        # Calculate reward
        reward = self.config.reward_goal if self.is_goal_state(resulting_state) else self.config.reward_step
        episode_terminated = self.is_goal_state(resulting_state)
        
        self.current_position = resulting_state
        return resulting_state, reward, episode_terminated
    
    def get_all_states(self) -> List[Tuple[int, int]]:
        """Retrieve all possible non-terminal states"""
        state_list = []
        for row in range(self.config.rows):
            for col in range(self.config.columns):
                if (row, col) != self.config.goal_position:
                    state_list.append((row, col))
        return state_list


class PolicyManager:
    """Advanced policy management with multiple selection strategies"""
    
    @staticmethod
    def select_epsilon_greedy_action(
        q_function: Dict, 
        state: Tuple[int, int], 
        available_actions: List[ActionDirection],
        exploration_rate: float
    ) -> ActionDirection:
        """Epsilon-greedy action selection with tie-breaking"""
        if random.random() < exploration_rate:
            return random.choice(available_actions)
        
        action_values = [q_function.get((state, action), 0.0) for action in available_actions]
        optimal_value = max(action_values)
        optimal_actions = [action for action, value in zip(available_actions, action_values) 
                          if value == optimal_value]
        return random.choice(optimal_actions)
    
    @staticmethod
    def select_greedy_action(
        q_function: Dict, 
        state: Tuple[int, int], 
        available_actions: List[ActionDirection]
    ) -> ActionDirection:
        """Greedy action selection with random tie-breaking"""
        action_values = [q_function.get((state, action), 0.0) for action in available_actions]
        optimal_value = max(action_values)
        optimal_actions = [action for action, value in zip(available_actions, action_values) 
                          if value == optimal_value]
        return random.choice(optimal_actions)


class EpisodeGenerator:
    """Generate episodes using specified policies"""
    
    @staticmethod
    def generate_episode_trajectory(
        environment: StochasticWindyGrid,
        policy_function,
        q_function: Optional[Dict] = None,
        exploration_rate: float = 0.1
    ) -> List[Tuple]:
        """
        Generate complete episode trajectory
        Returns: list of (state, action, reward) transitions
        """
        trajectory = []
        current_state = environment.reset_agent()
        
        while True:
            if q_function is not None:
                selected_action = policy_function(q_function, current_state, 
                                                environment.actions, exploration_rate)
            else:
                selected_action = policy_function(current_state, environment.actions)
            
            next_state, reward, terminal = environment.execute_action(selected_action)
            trajectory.append((current_state, selected_action, reward))
            
            if terminal:
                break
            current_state = next_state
        
        return trajectory


class DynamicProgrammingSolver:
    """
    Dynamic Programming Control using Policy Iteration
    with advanced convergence detection
    """
    
    def __init__(self, environment: StochasticWindyGrid, discount: float = 1.0):
        self.env = environment
        self.discount_factor = discount
        
    def solve_optimal_policy(self, tolerance: float = 1e-6, max_iterations: int = 1000):
        """Solve for optimal policy using policy iteration"""
        print("Executing Dynamic Programming Policy Iteration...")
        
        state_values = defaultdict(float)
        policy_function = {}
        all_states = self.env.get_all_states() + [self.env.config.goal_position]
        
        for iteration in range(max_iterations):
            max_change = 0
            
            # Policy Evaluation
            for state in all_states:
                if self.env.is_goal_state(state):
                    state_values[state] = 0
                    continue
                
                current_row, current_col = state
                action_values = []
                
                for action in self.env.actions:
                    expected_value = 0
                    wind_scenarios = [
                        (0.1, self.env.config.wind_pattern[current_col] + 1),
                        (0.8, self.env.config.wind_pattern[current_col]),
                        (0.1, 0)
                    ]
                    
                    for probability, wind_strength in wind_scenarios:
                        row_move, col_move = self.env.action_vectors[action]
                        potential_row = current_row + row_move - wind_strength
                        potential_col = current_col + col_move
                        
                        next_row, next_col = self.env._enforce_boundaries(potential_row, potential_col)
                        next_state = (next_row, next_col)
                        
                        transition_reward = (self.env.config.reward_goal 
                                           if self.env.is_goal_state(next_state) 
                                           else self.env.config.reward_step)
                        
                        expected_value += probability * (
                            transition_reward + self.discount_factor * state_values[next_state]
                        )
                    
                    action_values.append(expected_value)
                
                new_value = max(action_values)
                max_change = max(max_change, abs(state_values[state] - new_value))
                state_values[state] = new_value
            
            if max_change < tolerance:
                print(f"  Policy iteration converged in {iteration + 1} iterations")
                break
        
        # Extract Q-function from state values
        q_function = self._compute_q_function(state_values)
        return q_function, state_values
    
    def _compute_q_function(self, state_values: Dict) -> Dict:
        """Compute Q-function from state value function"""
        q_function = {}
        all_states = self.env.get_all_states() + [self.env.config.goal_position]
        
        for state in all_states:
            if self.env.is_goal_state(state):
                continue
            
            current_row, current_col = state
            for action in self.env.actions:
                q_value = 0
                wind_scenarios = [
                    (0.1, self.env.config.wind_pattern[current_col] + 1),
                    (0.8, self.env.config.wind_pattern[current_col]),
                    (0.1, 0)
                ]
                
                for probability, wind_strength in wind_scenarios:
                    row_move, col_move = self.env.action_vectors[action]
                    potential_row = current_row + row_move - wind_strength
                    potential_col = current_col + col_move
                    
                    next_row, next_col = self.env._enforce_boundaries(potential_row, potential_col)
                    next_state = (next_row, next_col)
                    
                    transition_reward = (self.env.config.reward_goal 
                                       if self.env.is_goal_state(next_state) 
                                       else self.env.config.reward_step)
                    
                    q_value += probability * (
                        transition_reward + self.discount_factor * state_values[next_state]
                    )
                
                q_function[(state, action)] = q_value
        
        return q_function


class MonteCarloOnPolicyLearner:
    """
    Monte Carlo On-Policy Control with first-visit method
    and incremental updates
    """
    
    def __init__(self, environment: StochasticWindyGrid, discount: float = 1.0):
        self.env = environment
        self.discount_factor = discount
        
    def learn_policy(self, episodes: int = 5000, exploration_rate: float = 0.1, 
                    learning_rate: float = 0.1):
        """Learn policy using on-policy Monte Carlo control"""
        print("Training Monte Carlo On-Policy Control...")
        
        q_function = defaultdict(float)
        visitation_counts = defaultdict(int)
        episode_lengths = []
        
        for episode_idx in range(episodes):
            trajectory = EpisodeGenerator.generate_episode_trajectory(
                self.env, PolicyManager.select_epsilon_greedy_action, 
                q_function, exploration_rate
            )
            episode_lengths.append(len(trajectory))
            
            visited_state_actions = set()
            cumulative_return = 0
            
            # Process episode in reverse for return calculation
            for step in reversed(range(len(trajectory))):
                state, action, reward = trajectory[step]
                cumulative_return = self.discount_factor * cumulative_return + reward
                
                if (state, action) not in visited_state_actions:
                    visited_state_actions.add((state, action))
                    visitation_counts[(state, action)] += 1
                    current_q = q_function[(state, action)]
                    q_function[(state, action)] = current_q + learning_rate * (
                        cumulative_return - current_q
                    )
            
            if (episode_idx + 1) % 500 == 0:
                avg_steps = np.mean(episode_lengths[-500:])
                print(f"  Episode {episode_idx + 1}/{episodes}, Average Steps: {avg_steps:.2f}")
        
        return q_function, episode_lengths


class MonteCarloOffPolicyLearner:
    """
    Monte Carlo Off-Policy Control with both weighted and unweighted
    importance sampling variants
    """
    
    def __init__(self, environment: StochasticWindyGrid, discount: float = 1.0):
        self.env = environment
        self.discount_factor = discount
        
    def learn_unweighted(self, episodes: int = 5000, exploration_rate: float = 0.1,
                        learning_rate: float = 0.1):
        """Off-policy control with unweighted importance sampling"""
        print("Training Monte Carlo Off-Policy Control (Unweighted)...")
        
        q_function = defaultdict(float)
        episode_lengths = []
        
        for episode_idx in range(episodes):
            trajectory = EpisodeGenerator.generate_episode_trajectory(
                self.env, PolicyManager.select_epsilon_greedy_action,
                q_function, exploration_rate
            )
            episode_lengths.append(len(trajectory))
            
            visited_pairs = set()
            cumulative_return = 0
            importance_ratio = 1.0
            
            for step in reversed(range(len(trajectory))):
                state, action, reward = trajectory[step]
                cumulative_return = self.discount_factor * cumulative_return + reward
                
                if (state, action) not in visited_pairs:
                    visited_pairs.add((state, action))
                    current_q = q_function[(state, action)]
                    q_function[(state, action)] = current_q + learning_rate * (
                        importance_ratio * (cumulative_return - current_q)
                    )
                
                # Update importance sampling ratio
                greedy_action = PolicyManager.select_greedy_action(
                    q_function, state, self.env.actions
                )
                
                if action == greedy_action:
                    target_prob = 1.0
                    behavior_prob = (exploration_rate / len(self.env.actions) + 
                                   (1 - exploration_rate))
                else:
                    target_prob = 0.0
                    behavior_prob = exploration_rate / len(self.env.actions)
                
                if target_prob == 0:
                    break
                
                importance_ratio *= target_prob / behavior_prob
            
            if (episode_idx + 1) % 500 == 0:
                avg_steps = np.mean(episode_lengths[-500:])
                print(f"  Episode {episode_idx + 1}/{episodes}, Average Steps: {avg_steps:.2f}")
        
        return q_function, episode_lengths
    
    def learn_weighted(self, episodes: int = 5000, exploration_rate: float = 0.1,
                      learning_rate: float = 0.1):
        """Off-policy control with weighted importance sampling"""
        print("Training Monte Carlo Off-Policy Control (Weighted)...")
        
        q_function = defaultdict(float)
        weight_accumulator = defaultdict(float)
        episode_lengths = []
        
        for episode_idx in range(episodes):
            trajectory = EpisodeGenerator.generate_episode_trajectory(
                self.env, PolicyManager.select_epsilon_greedy_action,
                q_function, exploration_rate
            )
            episode_lengths.append(len(trajectory))
            
            cumulative_return = 0
            importance_ratio = 1.0
            
            for step in reversed(range(len(trajectory))):
                state, action, reward = trajectory[step]
                cumulative_return = self.discount_factor * cumulative_return + reward
                
                weight_accumulator[(state, action)] += importance_ratio
                current_q = q_function.get((state, action), 0.0)
                
                update_step = (importance_ratio / weight_accumulator[(state, action)]) * (
                    cumulative_return - current_q
                )
                q_function[(state, action)] = current_q + update_step
                
                # Update importance ratio
                greedy_action = PolicyManager.select_greedy_action(
                    q_function, state, self.env.actions
                )
                
                if action != greedy_action:
                    break
                
                if action == greedy_action:
                    target_prob = 1.0
                    behavior_prob = (exploration_rate / len(self.env.actions) + 
                                   (1 - exploration_rate))
                else:
                    target_prob = 0.0
                    behavior_prob = exploration_rate / len(self.env.actions)
                
                importance_ratio *= target_prob / behavior_prob
            
            if (episode_idx + 1) % 500 == 0:
                avg_steps = np.mean(episode_lengths[-500:])
                print(f"  Episode {episode_idx + 1}/{episodes}, Average Steps: {avg_steps:.2f}")
        
        return q_function, episode_lengths


class TemporalDifferenceOnPolicyLearner:
    """
    TD(0) On-Policy Control (SARSA) with epsilon-greedy exploration
    """
    
    def __init__(self, environment: StochasticWindyGrid, discount: float = 1.0):
        self.env = environment
        self.discount_factor = discount
        
    def learn_policy(self, episodes: int = 5000, exploration_rate: float = 0.1,
                    learning_rate: float = 0.1):
        """Learn policy using SARSA algorithm"""
        print("Training TD(0) On-Policy Control (SARSA)...")
        
        q_function = defaultdict(float)
        episode_lengths = []
        
        for episode_idx in range(episodes):
            current_state = self.env.reset_agent()
            current_action = PolicyManager.select_epsilon_greedy_action(
                q_function, current_state, self.env.actions, exploration_rate
            )
            
            steps = 0
            while True:
                next_state, reward, terminal = self.env.execute_action(current_action)
                steps += 1
                
                if terminal:
                    # Update for terminal transition
                    td_error = reward - q_function[(current_state, current_action)]
                    q_function[(current_state, current_action)] += learning_rate * td_error
                    break
                
                # Select next action using policy
                next_action = PolicyManager.select_epsilon_greedy_action(
                    q_function, next_state, self.env.actions, exploration_rate
                )
                
                # SARSA update
                td_target = reward + self.discount_factor * q_function[(next_state, next_action)]
                td_error = td_target - q_function[(current_state, current_action)]
                q_function[(current_state, current_action)] += learning_rate * td_error
                
                current_state = next_state
                current_action = next_action
            
            episode_lengths.append(steps)
            
            if (episode_idx + 1) % 500 == 0:
                avg_steps = np.mean(episode_lengths[-500:])
                print(f"  Episode {episode_idx + 1}/{episodes}, Average Steps: {avg_steps:.2f}")
        
        return q_function, episode_lengths


class TemporalDifferenceOffPolicyLearner:
    """
    TD(0) Off-Policy Control (Q-Learning) with epsilon-greedy behavior policy
    """
    
    def __init__(self, environment: StochasticWindyGrid, discount: float = 1.0):
        self.env = environment
        self.discount_factor = discount
        
    def learn_policy(self, episodes: int = 5000, exploration_rate: float = 0.1,
                    learning_rate: float = 0.1):
        """Learn policy using Q-Learning algorithm"""
        print("Training TD(0) Off-Policy Control (Q-Learning)...")
        
        q_function = defaultdict(float)
        episode_lengths = []
        
        for episode_idx in range(episodes):
            current_state = self.env.reset_agent()
            
            steps = 0
            while True:
                # Behavior policy: epsilon-greedy
                current_action = PolicyManager.select_epsilon_greedy_action(
                    q_function, current_state, self.env.actions, exploration_rate
                )
                
                next_state, reward, terminal = self.env.execute_action(current_action)
                steps += 1
                
                if terminal:
                    td_error = reward - q_function[(current_state, current_action)]
                    q_function[(current_state, current_action)] += learning_rate * td_error
                    break
                
                # Target policy: greedy (max Q)
                max_next_q = max([q_function.get((next_state, action), 0.0) 
                                for action in self.env.actions])
                
                # Q-Learning update
                td_target = reward + self.discount_factor * max_next_q
                td_error = td_target - q_function[(current_state, current_action)]
                q_function[(current_state, current_action)] += learning_rate * td_error
                
                current_state = next_state
            
            episode_lengths.append(steps)
            
            if (episode_idx + 1) % 500 == 0:
                avg_steps = np.mean(episode_lengths[-500:])
                print(f"  Episode {episode_idx + 1}/{episodes}, Average Steps: {avg_steps:.2f}")
        
        return q_function, episode_lengths


class PerformanceVisualizer:
    """Advanced visualization for algorithm performance comparison"""
    
    @staticmethod
    def plot_learning_curves(results: Dict, title: str = "Algorithm Performance Comparison"):
        """Plot smoothed learning curves for all algorithms"""
        plt.figure(figsize=(14, 8))
        
        color_palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3F7CAC', '#6B8E23']
        
        for idx, (algorithm_name, step_data) in enumerate(results.items()):
            # Apply Gaussian smoothing for better visualization
            window_size = min(100, len(step_data) // 10)
            if len(step_data) > window_size:
                kernel = np.ones(window_size) / window_size
                smoothed_data = np.convolve(step_data, kernel, mode='valid')
                episodes = range(len(smoothed_data))
                plt.plot(episodes, smoothed_data, label=algorithm_name, 
                        color=color_palette[idx % len(color_palette)], linewidth=2.5, alpha=0.8)
        
        plt.xlabel('Training Episodes', fontsize=14, fontweight='bold')
        plt.ylabel('Steps per Episode', fontsize=14, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def display_policy_visualization(environment: StochasticWindyGrid, q_function: Dict):
        """Display learned policy as directional arrows"""
        action_symbols = {
            ActionDirection.NORTH: '↑',
            ActionDirection.SOUTH: '↓', 
            ActionDirection.WEST: '←',
            ActionDirection.EAST: '→'
        }
        
        policy_grid = [['' for _ in range(environment.config.columns)] 
                      for _ in range(environment.config.rows)]
        
        for row in range(environment.config.rows):
            for col in range(environment.config.columns):
                position = (row, col)
                if position == environment.config.goal_position:
                    policy_grid[row][col] = 'G'
                elif position == environment.config.start_position:
                    q_values = [q_function.get((position, action), 0.0) 
                               for action in environment.actions]
                    best_action = environment.actions[np.argmax(q_values)]
                    policy_grid[row][col] = f'S{action_symbols[best_action]}'
                else:
                    q_values = [q_function.get((position, action), 0.0) 
                               for action in environment.actions]
                    best_action = environment.actions[np.argmax(q_values)]
                    policy_grid[row][col] = action_symbols[best_action]
        
        print("\n" + "="*60)
        print("LEARNED POLICY VISUALIZATION")
        print("="*60)
        print("S = Start Position, G = Goal Position")
        print(f"Wind Pattern: {environment.config.wind_pattern}")
        print()
        
        for row in policy_grid:
            print('  '.join(f'{cell:>3}' for cell in row))
        print()


class ComprehensiveExperiment:
    """
    Main experiment runner for comprehensive algorithm comparison
    """
    
    def __init__(self):
        self.config = GridConfiguration()
        self.environment = StochasticWindyGrid(self.config)
        
    def execute_comparison_study(self):
        """Execute complete algorithm comparison study"""
        print("="*70)
        print("ADVANCED WINDY GRIDWORLD REINFORCEMENT LEARNING STUDY")
        print("="*70)
        
        print(f"\nExperiment Configuration:")
        print(f"  Grid Dimensions: {self.config.rows} × {self.config.columns}")
        print(f"  Start: {self.config.start_position}, Goal: {self.config.goal_position}")
        print(f"  Wind Pattern: {self.config.wind_pattern}")
        print(f"  Step Reward: {self.config.reward_step}, Goal Reward: {self.config.reward_goal}")
        
        # Training parameters
        training_episodes = 3000
        discount_factor = 1.0
        exploration_parameter = 0.1
        learning_step_size = 0.1
        
        results_data = {}
        learned_policies = {}
        
        print("\n" + "="*70)
        print("ALGORITHM TRAINING PHASE")
        print("="*70)
        
        # 1. Dynamic Programming Control
        dp_solver = DynamicProgrammingSolver(self.environment, discount_factor)
        q_dp, v_dp = dp_solver.solve_optimal_policy()
        learned_policies['DP Control'] = q_dp
        print()
        
        # 2. Monte Carlo On-Policy Control
        mc_on_learner = MonteCarloOnPolicyLearner(self.environment, discount_factor)
        q_mc_on, steps_mc_on = mc_on_learner.learn_policy(
            training_episodes, exploration_parameter, learning_step_size
        )
        results_data['MC On-Policy'] = steps_mc_on
        learned_policies['MC On-Policy'] = q_mc_on
        print()
        
        # 3. Monte Carlo Off-Policy Control (Unweighted)
        mc_off_learner = MonteCarloOffPolicyLearner(self.environment, discount_factor)
        q_mc_off_unw, steps_mc_off_unw = mc_off_learner.learn_unweighted(
            training_episodes, exploration_parameter, learning_step_size
        )
        results_data['MC Off-Policy (Unweighted)'] = steps_mc_off_unw
        learned_policies['MC Off-Policy (Unweighted)'] = q_mc_off_unw
        print()
        
        # 4. Monte Carlo Off-Policy Control (Weighted)
        q_mc_off_w, steps_mc_off_w = mc_off_learner.learn_weighted(
            training_episodes, exploration_parameter, learning_step_size
        )
        results_data['MC Off-Policy (Weighted)'] = steps_mc_off_w
        learned_policies['MC Off-Policy (Weighted)'] = q_mc_off_w
        print()
        
        # 5. TD(0) On-Policy Control (SARSA)
        td_on_learner = TemporalDifferenceOnPolicyLearner(self.environment, discount_factor)
        q_td_on, steps_td_on = td_on_learner.learn_policy(
            training_episodes, exploration_parameter, learning_step_size
        )
        results_data['TD(0) On-Policy (SARSA)'] = steps_td_on
        learned_policies['TD(0) On-Policy'] = q_td_on
        print()
        
        # 6. TD(0) Off-Policy Control (Q-Learning)
        td_off_learner = TemporalDifferenceOffPolicyLearner(self.environment, discount_factor)
        q_td_off, steps_td_off = td_off_learner.learn_policy(
            training_episodes, exploration_parameter, learning_step_size
        )
        results_data['TD(0) Off-Policy (Q-Learning)'] = steps_td_off
        learned_policies['TD(0) Off-Policy'] = q_td_off
        print()
        
        # Generate comprehensive visualizations
        print("="*70)
        print("PERFORMANCE ANALYSIS AND VISUALIZATION")
        print("="*70)
        PerformanceVisualizer.plot_learning_curves(results_data, 
            "Advanced Windy Gridworld: Control Algorithm Performance")
        
        # Display learned policy example
        PerformanceVisualizer.display_policy_visualization(self.environment, q_td_on)
        
        # Final performance summary
        print("="*70)
        print("FINAL PERFORMANCE SUMMARY (Last 100 Episodes)")
        print("="*70)
        for algorithm, steps in results_data.items():
            final_performance = np.mean(steps[-100:])
            improvement = ((steps[0] - final_performance) / steps[0]) * 100
            print(f"  {algorithm:35s}: {final_performance:6.2f} steps "
                  f"({improvement:5.1f}% improvement)")
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        return results_data, learned_policies


def main():
    """Main execution function"""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run comprehensive experiment
    experiment = ComprehensiveExperiment()
    results, policies = experiment.execute_comparison_study()
    
    return results, policies


if __name__ == "__main__":
    results, policies = main()