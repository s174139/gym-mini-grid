from perform_exp import perform_exp



if __name__ == "__main__":
    envn = 'CliffWalking-v0'
    envn = 'MiniGrid-Empty-5x5-v0'
    methods = ["sarsa", "sarsaL", "q_agent"]
    perform_exp(methods, envn)
