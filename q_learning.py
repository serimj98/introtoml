from environment import MountainCar
import sys
import numpy as np
import random

seed = 10601

if __name__ == "__main__":
    mode = sys.argv[1] #mode to run environment in (raw/tile)
    weight_out = sys.argv[2] #path to output weights of linear model
    returns_out = sys.argv[3] #path to output returns of agent
    episodes = int(sys.argv[4]) #number of episodes program should train agent for
    max_iterations = int(sys.argv[5]) #maximum of the length of an episode; when reached, terminate current episode
    epsilon = float(sys.argv[6]) #value for epsilon-greedy strategy
    gamma = float(sys.argv[7]) #discount factor
    learning_rate = float(sys.argv[8]) #learning rate of Q-learrning algorithm

mcar = MountainCar(mode)
bias_real = 0
actions = [0, 1, 2] #0 = pushing car left, 1 = doing nothing, 2 = pushing car right
weight_real = np.zeros((mcar.state_space, len(actions)))

def evaluate(action, state, weight, bias):
	total = bias
	for p, v in state.items():
		total += weight[:, action][p] * v
	return total

def policy(state, weight, bias, epsilon):
	if (random.random() >= epsilon): #exploit
		max_val = []
		for i in range(len(actions)):
			q = evaluate(i, state, weight, bias)
			max_val.append(q)
		max_q = max(max_val)
		action = max_val.index(max_q)
	else: #explore
		action = random.choice(actions)
	return action

def update(weight, bias, episodes, max_iterations, epsilon, gamma, learning_rate):
	returns = []
	for num in range(episodes):
		state = mcar.reset()
		done = False
		total_reward = []
		iteration_num = 0
		while (done == False): #do forever
			action = policy(state, weight, bias, epsilon) #select action a
			new_state, reward, done = mcar.step(action) #receive reward r, may change done state
			total_reward.append(reward)
			q_val = evaluate(action, state, weight, bias)
			max_val = []
			for i in range(len(actions)):
				q = evaluate(i, new_state, weight, bias)
				max_val.append(q)
			update = learning_rate * (q_val - (reward + (gamma * max(max_val)))) #derived update rule
			for p, v in state.items(): #update weight
				weight[p][action] -= update*v
			state = new_state #update state
			bias -= update #update bias
			iteration_num += 1 #keep track of max_iterations
			if (iteration_num >= max_iterations):
				break
		returns.append(sum(total_reward))
	return weight, bias, returns

weight, bias, returns = update(weight_real, bias_real, episodes, max_iterations, epsilon, gamma, learning_rate)

with open(weight_out, "w") as weight_out:
	weight_string = str(bias) + "\n"
	for i in weight:
		for j in i:
			weight_string += str(j) + "\n"
	print(weight_string, file=weight_out)

with open(returns_out, "w") as returns_out:
	return_string = ""
	for i in returns:
		return_string += str(i) + "\n"
	print(return_string, file=returns_out)