from dqn import *
import time

def main():
    env     = gym.make('MountainCar-v0')

    trials  = 1000
    trial_len = 500

    action = 0

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    dqn_agent.load_model('success.model')

    steps = []
    for trial in range(trials):
        cur_state = env.reset()
        cur_state = cur_state.reshape(1,2)

        startTime = time.time()

        for step in range(trial_len):
            # env.render()
            
            # if step % 4 == 0:
            action = dqn_agent.act(cur_state)

            new_state, reward, done, _ = env.step(action)

            # reward = reward if not done else -20
            new_state = new_state.reshape(1,2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            dqn_agent.replay()       # internally iterates default (prediction) model
            
            cur_state = new_state
            if done:
                dqn_agent.target_train() # iterates target model
                break

        endTime = time.time()
        print( endTime - startTime )

        if step >= 199:
            print('Failed to complete in trial {}'.format(trial))
            print('Epsilon: %f' % (dqn_agent.epsilon))
            if trial % 100 == 0:
                dqn_agent.save_model('dqn_trials/trial-{}.model'.format(trial))
        else:
            print('Completed in {} trials'.format(trial))
            dqn_agent.save_model('success.model')
            # break

if __name__ == '__main__':
    main()