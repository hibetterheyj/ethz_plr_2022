# ===============================================
#   Perception and learning for robotics 2022
#   Cart pole control using Q-learning
# ===============================================

import gym
import numpy as np
import random
import pickle
import math
import argparse
import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)

# Inspired by https://github.com/MattChanTK/ai-gym/blob/master/cart_pole/cart_pole_q_learning_4D.py
# and https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578
# with thanks to Inkyu Sa and Nicholas Lawrance

class Reinforce(object):

    def __init__(self, non_markov=False):
        self.env = gym.make('CartPole-v0')
        self.non_markov = non_markov

        # Hyperparameter definition
        self.total_tr_epi = 100000
        self.max_steps_tr = 100
        self.total_test_epi = 5
        self.max_steps_test = 500
        
        self.min_lr = 0.2
        self.gamma = 0.99   # Discounting rate
        # The larger the gamma, the smaller the discount. This means the learning agent cares more about the long term
        # reward. On the other hand, the smaller the gamma, the bigger the discount. This means our agent cares more
        # about the short term reward (the nearest cheese).

        # Exploitation params
        self.epsilon = 1.0  # Exploration rate, https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe
        self.max_epsilon = 1.0
        self.min_epsilon = 1e-2 
        self.dr = 1e-2  # Decay rate

        # Angle at which to fail the episode
        self.num_actions = self.env.action_space.n

        # Define the discretisation of the state space
        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2],
                             math.radians(50)]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2],
                             -math.radians(50)]
        if self.non_markov:
            self.buckets = (3, 12)          # If non-Markovian, just have x (cart position) and theta (pole orientation)
        else:
            self.buckets = (3, 6, 12, 24)   # (x, x', theta, theta') for discretisation

        # Create the Q-table
        self.Q = np.zeros(self.buckets + (self.num_actions,))   # Creating a Q-Table for each state-action pair

    def train(self, output_file=None):
        # Train a model by running the simulation using a policy and updating the Q-function
        scores = []
        self.mean_score = []
        print("Start training")
        lr = self.get_learning_rate(0)
        self.epsilon = self.get_explore_rate(0)

        # ==================
        #  Starting episode
        # ==================
        for episode in range(self.total_tr_epi):
            state = self.discretize(self.env.reset())
            tick_cnt = 0
            # =========================================================================
            #  Steps until either terminal conditions meet or max steps for training
            # =========================================================================
            for step in range(self.max_steps_tr):

                # To choose action
                action = self.get_action(state)

                # Once action selected, iterating
                observation, reward, done, info = self.env.step(action)
                new_state = self.discretize(observation)

                # Based on what we got, update Q value by making use of Bellman Eq.
                self.Q[state][action] = self.Q[state][action] + lr*(reward+self.gamma*np.max(self.Q[new_state]) - self.Q[state][action])

                # Update state
                tick_cnt += 1
                state = new_state
                if done:
                    break

            # Update epsilon
            self.epsilon = self.get_explore_rate(episode)
            lr = self.get_learning_rate(episode)
            if episode % 1000 == 0:
                print("Current episode = ", episode)
            scores.append(tick_cnt)
            if episode % self.max_steps_tr == 0:
                self.mean_score.append(np.mean(scores))
                print('[Episode {}] - Mean survival time over last {} episodes was {} ticks.'.format(episode, self.max_steps_tr, self.mean_score[-1]))
                scores = []

        if output_file is not None:
            self.save_Q(output_file)
            print("Model saved as ", output_file)

    def _get_bucket_index(self, ratios):
        new_obs = [int(round((bs - 1) * r)) for r, bs in zip(ratios, self.buckets)]
        new_obs = [min(bs - 1, max(0, o)) for o, bs in zip(new_obs, self.buckets)]
        return new_obs

    def discretize(self, obs):

        ratios = [(obs[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i]) for i in range(len(obs))]

        # Get the bucket indices for the current state
        if self.non_markov:
            new_obs = self._get_bucket_index([ratios[0], ratios[2]])  # extract cart position and pole orientation
        else:
            new_obs = self._get_bucket_index(ratios)
        return tuple(new_obs)

    def get_explore_rate(self,t):
        return max(self.min_epsilon, min(1.0, 1.0 - math.log10((t+1)/25)))

    def get_learning_rate(self, t):
        return max(self.min_lr, min(0.5, 1.0 - math.log10((t+1)/25)))
    
    def get_action(self,state):
        # Select a random action
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        # Select the action with the highest q
        else:
            action = np.argmax(self.Q[state])
        return action

    def test(self):
        # For testing
        print("Start testing")
        self.env.reset()
        rewards = []    # Reward history over episodes

        for episode in range(self.total_test_epi):
            state = self.env.reset()
            state = self.discretize(state)
            cumu_rewards = 0
            print("=======Episode======")
            for step in range(self.max_steps_test):
                action = np.argmax(self.Q[state])
                self.env.render()
                new_state, reward, done, info = self.env.step(action)
                cumu_rewards += reward
                state = self.discretize(new_state)
            rewards.append(cumu_rewards)
        self.env.close()
        print("Score over time: {}".format(sum(rewards)/self.total_test_epi))

    def load_Q(self, model_name):
        with open(model_name, "rb") as f:
            self.Q = pickle.load(f)
            self.buckets = self.Q.shape[:4]
            print("self.buckets=", self.buckets)
            print("Load Q table")

    def save_Q(self, model_name):
        with open(model_name, "wb") as f:
            pickle.dump(self.Q, f)
            print("Save Q table")

    def plot_Qslice(self):
        # (Very) Simple script for plotting a slice of the Q-function

        if self.non_markov:
            Qslice = self.Q[:, :, 0]
        else:
            plot_x_prime = 3
            plot_th_prime = 12
            plot_a = 0
            Qslice = self.Q[:, plot_x_prime, :, plot_th_prime, plot_a]
        f, a = plt.subplots()
        a.imshow(Qslice, origin='lower')
        a.set_xlabel('theta_i')
        a.set_ylabel('x_i')
        f.savefig('plotQ.pdf', bbox_inches='tight')

    def plot_reward(self):
        # Plot the reward over episodes
        f, a = plt.subplots()
        a.plot(self.mean_score)
        a.set_xlabel('Episode number (x{})'.format(self.max_steps_tr))
        a.set_ylabel('Mean survival time over last {} episodes'.format(self.max_steps_tr))
        f.savefig('training.pdf', bbox_inches='tight')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train or test a simple Q-learning agent on the OpenAI gym CartPole environment')
    parser.add_argument('-train', type=str, default='policy0.pickle', help='Specify a target file for a trained policy')
    parser.add_argument('-test', type=str, default=None, help='Specify a trained policy (pickle file) and test it')
    parser.add_argument('-non_markov', action='store_true', help='Use a reduced (non-Markovian) state space')
    args = parser.parse_args()

    myReinforce = Reinforce(non_markov=args.non_markov)

    if args.test is not None:
        myReinforce.load_Q(args.test)
        myReinforce.test()

    else:
        myReinforce.train(output_file=args.train)
        myReinforce.plot_reward()
