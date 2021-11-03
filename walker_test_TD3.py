import gym
from TD3 import TD3
import time


def test():
    env_name = "Walker2d-v2"
    random_seed = 4
    run_version = 4
    n_episodes = 5
    lr = 0.002
    max_timesteps = 1000
    render = True
    frame_delay = 0.01


    filename = "TD3_{}_{}_{}".format(env_name, random_seed, run_version)
    directory = "./TD3_preTrained"

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(lr, state_dim, action_dim, max_action)

    policy.load_actor(directory, filename)

    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
                time.sleep(frame_delay)
            if done:
                break

        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
    env.close()


if __name__ == '__main__':
    test()