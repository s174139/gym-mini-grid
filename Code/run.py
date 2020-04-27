from agents import *


# path = 'D:/OneDrive/Dokumenter/DTU/4. Sem/Intro til Reinforcement learning/Project/Gym_mini_grid/Code'
# import sys
# sys.path.append(path)


q_exp = f"experiments/cliffwalk_Q"
def cliffwalk():
    env = gym.make('CliffWalking-v0')
    agent = QAgent(env, epsilon=0.1, alpha=0.5)
    train(env, agent, q_exp, num_episodes=200, max_runs=10)
    return env, q_exp
    
if __name__ == "__main__":
    env, exp_name = cliffwalk()
    main_plot(exp_name, smoothing_window=10)
    plt.ylim([-100, 0])
    plt.title("Q-learning on " + env.spec._env_name)
    savepdf("Q_learning_cliff")
    plt.show()


# sarsa_exp = f"experiments/cliffwalk_Sarsa"
# if __name__ == "__main__":
#     env, q_experiment = cliffwalk()  # get results from Q-learning
#     agent = SarsaAgent(env, epsilon=0.1, alpha=0.5)
#     train(env, agent, sarsa_exp, num_episodes=200, max_runs=10)
#     main_plot([q_experiment, sarsa_exp], smoothing_window=10)
#     plt.ylim([-100, 0])
#     plt.title("Q and Sarsa learning on " + env.spec._env_name)
#     savepdf("QSarsa_learning_cliff")
#     plt.show()


# if __name__ == "__main__":
#     envn = 'CliffWalking-v0'
#     env = gym.make(envn)

#     # methods = ["MC", "Q", "Sarsa"]
#     alpha =0.05
#     sarsaLagent = SarsaLambdaAgent(env,gamma=0.99, epsilon=0.1, alpha=alpha, lamb=0.9)
#     sarsa = SarsaAgent(env,gamma=0.99,alpha=alpha,epsilon=0.1)
#     methods = [("SarsaL", sarsaLagent), ("Sarsa", sarsa)]

#     experiments = []
#     for k, (name,agent) in enumerate(methods):
#         expn = f"experiments/{envn}_{name}"
#         train(env, agent, expn, num_episodes=500, max_runs=10)
#         experiments.append(expn)
#     main_plot(experiments, smoothing_window=10, resample_ticks=200)
#     plt.ylim([-100, 0])
#     savepdf("cliff_sarsa_lambda")
#     plt.show()
