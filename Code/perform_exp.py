from agents import *
# import gym
from gym_minigrid.wrappers import *
import shutil
import os.path

def perform_exp(methods, envn, gamma=0.99, epsilon=0.1, alpha=0.05, lamb=0.9, del_prev = False, n_runs = 1): 

    env = gym.make(envn)

    method_names = list(methods) if type(methods) != list else methods

    # methods = ["MC", "Q", "Sarsa"]

    methods = []
    if "sarsaL" in method_names:
        sarsaLagent = SarsaLambdaAgent(env,gamma=gamma, epsilon=epsilon, alpha=alpha, lamb=lamb)
        methods.append(("SarsaL", sarsaLagent))
    if "sarsa" in method_names:
        sarsa = SarsaAgent(env,gamma=gamma,alpha=alpha,epsilon=epsilon)
        methods.append(("Sarsa", sarsa))
    if "q_agent" in method_names:
        q_agent = QAgent(env, epsilon=epsilon, alpha=alpha)
        methods.append(("Q-learn", q_agent))
    # methods = [("SarsaL", sarsaLagent), ("Sarsa", sarsa), ("Q-learn", q_agent)]
    print(methods)
    experiments = []
    for k, (name,agent) in enumerate(methods):
        expn = f"experiments/{envn}_{name}"
        if del_prev & os.path.exists(expn):
            shutil.rmtree(expn)
        train(env, agent, expn, num_episodes=500, max_runs=10)
        experiments.append(expn)
    if n_runs > 1:
        perform_exp(method_names, envn, gamma, epsilon, alpha, lamb, del_prev = False, n_runs = n_runs-1)
    else:
        main_plot(experiments, smoothing_window=10, resample_ticks=200)
        plt.ylim([-100, 0])
        savepdf(envn)
        plt.show()