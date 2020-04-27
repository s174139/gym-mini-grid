import sys
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from collections import OrderedDict, defaultdict
import os
import glob
import csv
from gym.envs.toy_text.discrete import DiscreteEnv
import gym

from __init__ import log_time_series, main_plot, savepdf
from utils.common import defaultdict2
from utils.irlc_plot import existing_runs

class Agent(): 
    def __init__(self, env, gamma=0.99, epsilon=0):
        self.env, self.gamma, self.epsilon = env, gamma, epsilon 
        self.Q = defaultdict2(lambda s: np.zeros(len(env.P[s]) if hasattr(env, 'P') and s in env.P else env.action_space.n))
    def pi(self, s): 
        """ Should return the Agent's action in s (i.e. an element contained in env.action_space)"""
        raise NotImplementedError("return action") 

    def train(self, s, a, r, sp, done=False): 
        """ Called at each step of the simulation.
        The agent was in s, took a, ended up in sp (with reward r) and done indicates if the environment terminated """
        raise NotImplementedError() 

    def __str__(self):
        warnings.warn("Please implement string method for caching; include ALL parameters")
        return super().__str__()

    def random_pi(self, s):
        """ Generates a random action given s.

        It might seem strange why this is useful, however many policies requires us to to random exploration.
        We will implement the method depending on whether self.env defines an MDP or just contains an action space.
        """
        if isinstance(self.env, DiscreteEnv):
            return np.random.choice(list(self.env.P[s].keys()))
        else:
            return self.env.action_space.sample()

    def pi_eps(self, s): 
        """ Implement epsilon-greedy exploration. Return random action with probability self.epsilon,
        else be greedy wrt. the Q-values. """
        #return np.random.choice(len(self.Q[s])) if np.random.random() < self.epsilon else np.argmax(self.Q[s])

        return np.argmax(self.Q[s]) if np.random.binomial(1, p = 1-self.epsilon) == 1 else np.random.choice(len(self.Q[s]))
        raise NotImplementedError("Implement function body")

    def value(self, s):
        return np.max(self.Q[s])

class ValueAgent(Agent): 
    def __init__(self, env, gamma=0.95, policy=None, v_init_fun=None):
        self.env = env
        self.policy = policy  # policy to evaluate 
        """ Value estimates. 
        Initially v[s] = 0 unless v_init_fun is given in which case v[s] = v_init_fun(s). """
        self.v = defaultdict2(float if v_init_fun is None else v_init_fun)
        super().__init__(env, gamma)

    def pi(self, s):  
        return self.random_pi(s) if self.policy is None else self.policy(s)

    def value(self, s):
        return self.v[s] 

def load_time_series(experiment_name):
    files = list(filter(os.path.isdir, glob.glob(experiment_name+"/*")))
    recent = sorted(files, key=lambda file: os.path.basename(file))[-1]
    stats = []
    with open(recent + '/log.txt', 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(csv_reader):
            if i == 0:
                head = row
            else:
                stats.append( {k:float(v) for k, v in zip(head, row) } )
    return stats, recent

def train(env, agent, experiment_name=None, num_episodes=None, verbose=True, reset=True, max_steps=1e10,
          max_runs=None, saveload_model=False):

    if max_runs is not None and existing_runs(experiment_name) >= max_runs:
            return experiment_name, None, True
    stats = []
    steps = 0
    ep_start = 0
    if saveload_model:  # Code for loading/saving models
        did_load = agent.load(os.path.join(experiment_name))
        if did_load:
            stats, recent = load_time_series(experiment_name=experiment_name)
            ep_start, steps = stats[-1]['Episode']+1, stats[-1]['Steps']

    done = False
    with tqdm(total=num_episodes, disable=not verbose) as tq:
        for i_episode in range(num_episodes): 
            s = env.reset() if reset else (env.s if hasattr(env, "s") else env.env.s) 
            reward = []
            for _ in itertools.count():
                a = agent.pi(s)
                sp, r, done, _ = env.step(a)
                agent.train(s, a, r, sp, done)
                reward.append(r)
                steps += 1
                if done or steps > max_steps:
                    break
                s = sp 

            stats.append({"Episode": i_episode + ep_start,
                          "Accumulated Reward": sum(reward),
                          "Average Reward": np.mean(reward),
                          "Length": len(reward),
                          "Steps": steps})
            tq.set_postfix(ordered_dict=OrderedDict(stats[-1]))
            tq.update()
    sys.stderr.flush()
    if saveload_model:
        agent.save(experiment_name)
        if did_load:
            os.rename(recent+"/log.txt", recent+"/log2.txt")  # Shuffle old logs

    if experiment_name is not None:
        log_time_series(experiment=experiment_name, list_obs=stats)
        print(f"Training completed. Logging: '{', '.join( stats[0].keys()) }' to {experiment_name}")
    return experiment_name, stats, done

class QAgent(Agent):
    """
    Implement the Q-learning agent here. Note that the Q-datastructure already exist
    (see agent class for more information)
    """
    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1):
        self.alpha = alpha
        super().__init__(env, gamma, epsilon)

    def pi(self, s): 
        """
        Return current action using epsilon-greedy exploration. Look at the Agent class
        for ideas.
        """
        return Agent.pi_eps(self, s)
        raise NotImplementedError("Implement function body")

    def train(self, s, a, r, sp, done=False): 
        """
        Implement the Q-learning update rule, i.e. compute a* from the Q-values.
        As a hint, note that self.Q[sp][a] corresponds to q(s_{t+1}, a) and
        that what you need to update is self.Q[s][a] = ...
        """
        # TODO: 2 lines missing.
        # ap = self.pi(s)
        self.Q[s][a] += self.alpha * (r + self.gamma*np.max(self.Q[sp])-self.Q[s][a])

        
        #raise NotImplementedError("Implement function body")

    def __str__(self):
        return f"QLearner_{self.gamma}_{self.epsilon}_{self.alpha}"

class SarsaAgent(QAgent):
    def __init__(self, env, gamma=0.99, alpha=0.5, epsilon=0.1):
        self.t = 0 # indicate we are at the beginning of the episode
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)

    def pi(self, s):
        if self.t == 0: 
            """ we are at the beginning of the episode. Generate a by being epsilon-greedy"""
            # TODO: 1 lines missing.
            return self.pi_eps(s)
            raise NotImplementedError("Implement function body")
        else: 
            """ Return the action self.a you generated during the train where you know s_{t+1} """
            # TODO: 1 lines missing.
            return self.a

            raise NotImplementedError("Implement function body")

    def train(self, s, a, r, sp,done=False):
        """
        generate A' as self.a by being epsilon-greedy. Re-use code from the Agent class.
        """
        # TODO: 1 lines missing.
        self.a = self.pi_eps(sp) if not done else -1

        #raise NotImplementedError("self.a = ....")
        """ now that you know A' = self.a, perform the update to self.Q[s][a] here """
        # TODO: 2 lines missing.
        delta = r + (self.gamma * self.Q[sp][self.a] if not done else 0) - self.Q[s][a] #!b
        self.Q[s][a] += self.alpha * delta #!b
        self.t = 0 if done else self.t + 1 # update current iteration number

    def __str__(self):
        return f"Sarsa{self.gamma}_{self.epsilon}_{self.alpha}"


class SarsaLambdaAgent(SarsaAgent):
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.5, lamb=0.9):
        """
        Implementation of Sarsa(Lambda) in the tabular version, see
        http://incompleteideas.net/book/first/ebook/node77.html
        for details (and note that as mentioned in the exercise description/lecture Sutton forgets to reset the
        eligibility trace after each episode).
        Note 'lamb' is an abbreveation of lambda, because lambda is a reserved keyword in python.

        The constructor initializes e, the eligibility trace, as a datastructure similar to self.Q. I.e.
        self.e[s][a] is the eligibility trace e(s,a).

        Since Sarsa(Lambda) generalize Sarsa, we have to generate the next action A' from S' in the train method and
        store it for when we take actions. I.e. we can re-use the Sarsa Agents code for acting (self.pi).
        """
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
        self.lamb = lamb
        self.e = defaultdict2(self.Q.default_factory)

    def train(self, s, a, r, sp, done=False):
        # TODO: 1 lines missing.
        # ap = self.pi(sp)
        ap = self.pi_eps(sp)
        #raise NotImplementedError("a_prime = ... (get action for S'=sp using self.pi_eps; see Sarsa)")
        
        # TODO: 1 lines missing.
        #raise NotImplementedError("delta = ... (The ordinary Sarsa learning signal)")
        delta = r + self.gamma*self.Q[sp][ap] - self.Q[s][a]

        # TODO: 1 lines missing.
        #raise NotImplementedError("Update the eligibility trace e(s,a) += 1")
        self.e[s][a] += 1

        for s, es in self.e.items():
            for a, e_sa in enumerate(es):
                # TODO: 2 lines missing.
                #raise NotImplementedError("Update Q values and eligibility trace")
                # if e_sa == 0:
                #     continue
                # else:
                self.Q[s][a] += self.alpha*delta*e_sa
                self.e[s][a] *= self.gamma*self.lamb

        if done: # Clear eligibility trace after each episode (missing in pseudo code) and update variables for Sarsa
            self.e.clear()
        else:
            self.a = ap
            self.t += 1

    def __str__(self):
        return f"SarsaLambda_{self.gamma}_{self.epsilon}_{self.alpha}_{self.lamb}"