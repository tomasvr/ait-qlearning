import simple_grid
from q_learning_skeleton import *
import gym

def act_loop(env, agent, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        print('---episode %d---' % episode)

        renderit = False
        if episode % 100 == 0:
           renderit = True

        printing = False
        agent.printEpsilon(episode) 
        for t in range(MAX_EPISODE_LENGTH):
            if renderit:
                env.render()

#            printing = True
            if t % 500 == 499:
                printing = True

            if printing:
                print('EPISODE: %d ---stage %d---' % (episode, t))
                agent.report()

            action = agent.select_action(state, episode)
            new_state, reward, done, info = env.step(action)
            if printing:
                print("act:", action)
                print("reward=%s" % reward)

            agent.process_experience(state, action, new_state, reward, done)
            state = new_state
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                env.render()
                agent.report()
                break
    env.close()


if __name__ == "__main__":
    env = simple_grid.DrunkenWalkEnv(map_name="walkInThePark")
    # env = simple_grid.DrunkenWalkEnv(map_name="theAlley")
    num_a = env.action_space.n

    if (type(env.observation_space)  == gym.spaces.discrete.Discrete):
        num_o = env.observation_space.n
    else:
        raise("Qtable only works for discrete observations")

    ql = QLearner(env) # can alter parameters here
    act_loop(env, ql, NUM_EPISODES)
    print(ql.q_table)


