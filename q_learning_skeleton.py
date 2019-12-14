import random
import numpy as np
import simple_grid

NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500

DISCOUNT = 0.9
LEARNINGRATE = 0.1


ENABLE_DECAY = True

# Only used if ENABLE_DECAY is disabled
STATIC_EPSILON = 0.05

# Only used if ENABLE_DCEAY is enabled
MAX_EPSILON = 1
MIN_EPSILON = 0.1
DECAY_EPSILON = 1

"""
    "DOCUMENTTION" LINKS:
    -(action) Spaces : https://github.com/openai/gym/blob/1bf4ae955b7064609821b331351083fa53cd5b66/gym/spaces/tuple.py#L41
    -Environments : https://github.com/openai/gym/blob/master/gym/core.py

"""

class QLearner():
    """
    Q-learning agent

    """

    #WARNING: edit main fn! Signiture changed to take in environment first!
    def __init__(self, env):
        # Initialize class properties #
        self.env = env
        self.name = "Agent_R1"
        self.discount = DISCOUNT
        self.learning_rate = LEARNINGRATE  #learning_rate = alpha
        self.epsilon = 1 # just to initalize

      	# Initalize Q table #
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n]); #Stores Q(s,a) for all s,a

    def process_experience(self, state, action, next_state, reward, done): 

        # Update q_table #
        if (not done): #TODO: is this check correct? -"epsiode terminates" == done ?
            self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
        	self.learning_rate * (reward + self.discount * np.max(self.q_table[next_state, :])) # \ chars needed when breaking lines
        else:
            self.q_table[state, action] = (1-self.learning_rate) * self.q_table[state, action] + \
        	self.learning_rate * (reward)

    # Returns an action based on current state
    def select_action(self, state, episode):
        if(ENABLE_DECAY):
            normalized_episode = (episode / NUM_EPISODES)
            self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_EPSILON * normalized_episode)
        else:
            self.epsilon = STATIC_EPSILON
#        print("epsilon: %f" % epsilon)
        # "Roll dice" to check whether to attempt random action #
        p = random.random()

        if (p <= self.epsilon):
            # Random action:
            random_action = self.env.action_space.sample() #A random action is sampled from the action space
            return random_action
        else:
            # Greedy action: Calculate max{a}(Q_old(s,a)) from table of Q_values
#            print("picking action")
#            print(self.q_table[state,:])
            greedy_action = np.argmax(self.q_table[state,:])
            return greedy_action
        pass

    # Function to print during episodes
    def report(self):
        print(self.name)
        print("Agent Report:")

    def printEpsilon(self):
        print("epsilon: %f" % self.epsilon)
