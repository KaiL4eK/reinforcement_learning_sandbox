from dqn import *
import time

def main():
    env     = gym.make('MountainCar-v0')

    dqn_agent = DQN(env=env, rand_act_prob=0.0)
    dqn_agent.load_model('success.model')

    cur_state = env.reset()

    while True:
        env.render()
        cur_state = cur_state.reshape(1,2)
        action = dqn_agent.act(cur_state)
        cur_state, reward, done, _ = env.step(action)

        if done:
            break

if __name__ == '__main__':
    main()