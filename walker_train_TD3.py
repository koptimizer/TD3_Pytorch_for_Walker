import torch
import gym
import numpy as np
from TD3 import TD3
from utils import ReplayBuffer
import random
import os
from datetime import datetime

def train():
    ######### Hyperparameters #########
    env_name = "Walker2d-v2"
    log_interval = 10  # print avg reward after interval
    random_seed = 4
    run_version = 4 # version
    gamma = 0.99  # discount for future rewards
    batch_size = 1024  # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = 0.1
    polyak = 0.995  # target policy update parameter (1-tau)
    policy_noise = 0.25  # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2  # delayed policy updates parameter
    max_episodes = 350  # max num of episodes
    max_timesteps = 1000  # max timesteps in one episode
    ###################################

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        if random_seed == 0:
            random_seed = random.randint(0, 1e4)
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    save_directory = "TD3_preTrained"  # save trained models
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    log_directory = "TD3_logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    filename = "TD3_{}_{}_{}".format(env_name, random_seed, run_version)
    save_model_freq = 50

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at GMT : ", start_time)
    print("============================================================================================")


    # logging variables:
    avg_reward = 0
    ep_reward = 0
    log_f = open(log_directory + "/TD3_" + env_name + "_log_" + str(run_version) + ".txt", "w+")

    # training procedure:
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            action = action.clip(env.action_space.low, env.action_space.high)

            # take action in env:
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state

            avg_reward += reward
            ep_reward += reward

            # if episode is done then update policy:
            if done or t == (max_timesteps - 1):
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                break

        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0

        if episode % save_model_freq == 0:
            policy.save(save_directory, filename)
            print("model save!")
            log_end_time = datetime.now().replace(microsecond=0)
            print("Training time log : {}".format(log_end_time - start_time))


        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = int(avg_reward / log_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()