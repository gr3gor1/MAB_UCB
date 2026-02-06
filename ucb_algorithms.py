import numpy as np
from scipy.stats import norm

class UCB1:
    def __init__(self, arms):

        self.arms = arms
        self.n_arms = len(arms)
        self.arm_to_index = {arm: idx for idx, arm in enumerate(arms)}
        self.index_to_arm = {idx: arm for idx, arm in enumerate(arms)}

        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.total_steps = 0

    def select_arm(self):
        self.total_steps += 1
        ucb_values = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return self.index_to_arm[i]
            average_reward = self.values[i]
            confidence = np.sqrt((2 * np.log(self.total_steps)) / self.counts[i])
            ucb_values[i] = average_reward + confidence

        selected_index = np.argmax(ucb_values)
        return self.index_to_arm[selected_index]

    def update(self, arm, reward):
        index = self.arm_to_index[arm]
        self.counts[index] += 1
        n = self.counts[index]
        value = self.values[index]
        self.values[index] += (reward - value) / n

class BayesUCB:

    def __init__(self, arms, mu0=0.0, lambda0=1e-6, alpha0=1.0, beta0=1.0, alpha_quant=1.0):
        self.arms = arms
        self.n_arms = len(arms)

        self.mu0 = mu0
        self.lambda0 = lambda0
        self.alpha0 = alpha0
        self.beta0 = beta0

        self.alpha_quant = alpha_quant

        self.counts = np.zeros(self.n_arms)
        self.sum_rewards = np.zeros(self.n_arms)
        self.sum_sq_rewards = np.zeros(self.n_arms)

        self.total_steps = 0

        self.arm_to_index = {arm: i for i, arm in enumerate(arms)}
        self.index_to_arm = {i: arm for i, arm in enumerate(arms)}


    def _posterior_params(self, i):
        n = self.counts[i]
        S = self.sum_rewards[i]
        Sq = self.sum_sq_rewards[i]

        lambda_n = self.lambda0 + n
        mu_n = (self.lambda0 * self.mu0 + S) / lambda_n

        alpha_n = self.alpha0 + 0.5 * n
        beta_n = self.beta0 + 0.5 * (Sq + self.lambda0 * self.mu0**2 - lambda_n * mu_n**2)

        return mu_n, lambda_n, alpha_n, beta_n


    def select_arm(self):
        self.total_steps += 1

        # Bayes-UCB quantile rule from the paper
        q_t = self._quantile()
        z = norm.ppf(q_t)

        ucb_values = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return self.index_to_arm[i]

            mu_n, lambda_n, alpha_n, beta_n = self._posterior_params(i)

            # sigma_post^2 = beta_n / ((alpha_n - 1) * lambda_n)
            sigma_post = np.sqrt(beta_n / ((alpha_n - 1) * lambda_n))

            ucb_values[i] = mu_n + z * sigma_post

        selected_index = np.argmax(ucb_values)
        return self.index_to_arm[selected_index]


    def update(self, arm, reward):
        i = self.arm_to_index[arm]
        self.counts[i] += 1
        self.sum_rewards[i] += reward
        self.sum_sq_rewards[i] += reward * reward

    def _quantile(self):
      t = self.total_steps
      if t < 3:
        return 0.95
      return 1 - 1.0 / (t * (np.log(t) ** self.alpha_quant))

class UCB_Tuned:
    def __init__(self, arms):
        self.arms = arms
        self.n_arms = len(arms)
        self.arm_to_index = {arm: idx for idx, arm in enumerate(arms)}
        self.index_to_arm = {idx: arm for idx, arm in enumerate(arms)}

        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.sum_sq_rewards = np.zeros(self.n_arms)
        self.total_steps = 0

        self.global_min_reward = np.inf
        self.global_max_reward = -np.inf

    def _scale_reward(self, reward):
        self.global_min_reward = min(self.global_min_reward, reward)
        self.global_max_reward = max(self.global_max_reward, reward)

        if self.global_max_reward == self.global_min_reward:
            return 0.5
        return (reward - self.global_min_reward) / (self.global_max_reward - self.global_min_reward)

    def select_arm(self):
        self.total_steps += 1
        ucb_values = np.zeros(self.n_arms)

        if np.any(self.counts == 0):
            return self.index_to_arm[np.argmin(self.counts)]

        log_t = np.log(self.total_steps)

        for i in range(self.n_arms):
            n_i = self.counts[i]
            mu_i = self.values[i]

            V_i = (self.sum_sq_rewards[i] / n_i) - (mu_i ** 2)

            inner_term = V_i + np.sqrt((2 * log_t) / n_i)

            V_tilde = np.minimum(inner_term, 0.25)

            ucb_values[i] = mu_i + np.sqrt((log_t / n_i) * V_tilde)

        selected_index = np.argmax(ucb_values)
        return self.index_to_arm[selected_index]

    def update(self, arm, reward):

        index = self.arm_to_index[arm]
        self.counts[index] += 1
        n = self.counts[index]

        old_mean = self.values[index]
        self.values[index] += (reward - old_mean) / n

        self.sum_sq_rewards[index] += reward ** 2

class UCB_V:
    def __init__(self, arms, B=1.0):
        self.arms = arms
        self.n_arms = len(arms)
        self.arm_to_index = {arm: idx for idx, arm in enumerate(arms)}
        self.index_to_arm = {idx: arm for idx, arm in enumerate(arms)}

        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.sum_sq_diff = np.zeros(self.n_arms)
        self.total_steps = 0
        self.B = B

        self.global_min_reward = np.inf
        self.global_max_reward = -np.inf

    def _scale_reward(self, reward):
        self.global_min_reward = min(self.global_min_reward, reward)
        self.global_max_reward = max(self.global_max_reward, reward)

        if self.global_max_reward == self.global_min_reward:
            return 0.5
        return (reward - self.global_min_reward) / (self.global_max_reward - self.global_min_reward)

    def select_arm(self):
        self.total_steps += 1
        ucb_values = np.zeros(self.n_arms)

        if np.any(self.counts == 0):
            return self.index_to_arm[np.argmin(self.counts)]

        log_t = np.log(self.total_steps)

        for i in range(self.n_arms):
            n_i = self.counts[i]
            mu_i = self.values[i]

            if n_i > 1:
                variance_i = self.sum_sq_diff[i] / (n_i)
            else:
                variance_i = 1e-9

            term1 = np.sqrt((2 * variance_i * log_t) / n_i)

            term2 = (3 * self.B * log_t) / n_i


            ucb_values[i] = mu_i + term1 + term2

        selected_index = np.argmax(ucb_values)
        return self.index_to_arm[selected_index]

    def update(self, arm, reward):

        index = self.arm_to_index[arm]
        self.counts[index] += 1
        n = self.counts[index]

        old_mean = self.values[index]

        self.values[index] += (reward - old_mean) / n

        new_mean = self.values[index]
        self.sum_sq_diff[index] += (reward - old_mean) * (reward - new_mean)

class UCB_BwK:
    def __init__(self, thresholds):
        self.arms = thresholds
        self.n_arms = len(thresholds)
        self.arm_to_index = {arm: idx for idx, arm in enumerate(thresholds)}
        self.index_to_arm = {idx: arm for idx, arm in enumerate(thresholds)}

        self.counts = np.zeros(self.n_arms)
        self.mu_rewards = np.zeros(self.n_arms)
        self.mu_costs = np.zeros(self.n_arms)
        self.total_steps = 0

    def select_arm(self):
        self.total_steps += 1
        ucb_ratio_values = np.zeros(self.n_arms)

        if np.any(self.counts == 0):
            return self.index_to_arm[np.argmin(self.counts)]

        log_t = np.log(self.total_steps)

        for i in range(self.n_arms):
            n_i = self.counts[i]

            confidence_ratio = np.sqrt((2 * log_t) / n_i)

            ucb_rew = self.mu_rewards[i] + confidence_ratio

            tau_t = np.sqrt(np.log(self.total_steps) / self.total_steps)
            ucb_cost = max(self.mu_costs[i] - confidence_ratio, tau_t)

            ucb_ratio_values[i] = ucb_rew / ucb_cost

        selected_index = np.argmax(ucb_ratio_values)
        return self.index_to_arm[selected_index]

    def update(self, arm, reward, actual_cost):
        index = self.arm_to_index[arm]
        self.counts[index] += 1
        n = self.counts[index]

        old_mu_reward = self.mu_rewards[index]
        self.mu_rewards[index] += (reward - old_mu_reward) / n

        old_mu_cost = self.mu_costs[index]
        self.mu_costs[index] += (actual_cost - old_mu_cost) / n

