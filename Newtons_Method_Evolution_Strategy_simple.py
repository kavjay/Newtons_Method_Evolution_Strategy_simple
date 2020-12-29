import numpy as np
import matplotlib.pyplot as plt
import time

time_seed = int(time.time())

# ORIGINAL EVOLUTION STRATEGY
def evolution_strategy(
    f,
    population_size,
    sigma,
    lr,
    initial_params,
    num_iters):

  np.random.seed(time_seed) # to keep both test cases consistant

  # assume initial params is a 1-D array
  num_params = len(initial_params)
  reward_per_iteration = np.zeros(num_iters)

  params = initial_params
  for t in range(num_iters):
    N = np.random.randn(population_size, num_params)
    R = np.zeros(population_size) # stores the reward

    # loop through each "offspring"
    for j in range(population_size):
      params_try = params + sigma*N[j]
      R[j] = f(params_try)

    m = R.mean()
    A = (R - m) / R.std()
    reward_per_iteration[t] = m
    params = params + lr/(population_size*sigma) * np.dot(N.T, A)

  return params, reward_per_iteration


# NEWTON'S METHOD BASED EVOLUTION STRATEGY
def newtons_method_evolution_strategy(
    f,
    population_size,
    sigma,
    lr,
    initial_params,
    num_iters):

  np.random.seed(time_seed) # to keep both test cases consistant

  # assume initial params is a 1-D array
  num_params = len(initial_params)
  reward_per_iteration = np.zeros(num_iters)

  params = initial_params
  for t in range(num_iters):

    N = np.random.randn(population_size, num_params)
    R = np.zeros(population_size) # stores the reward
    r0 = f(params)

    # loop through each "offspring"
    for j in range(population_size):
      params_try = params + sigma*N[j]
      R[j] = f(params_try)

    m = R.mean()
    A = (R - m) / R.std()
    reward_per_iteration[t] = m
    newtons_term = m - r0
    # do the Newton update term if the Newton's term is not zero, else do standard ES
    update_term = np.clip(-0.5*sigma*np.dot(N.T, R)/population_size/newtons_term, -1,1) if newtons_term != 0 else lr/(population_size*sigma) * np.dot(N.T, A)
    params = params + update_term

  return params, reward_per_iteration


def reward_function(params):
  x0 = params[0]
  x1 = params[1]
  x2 = params[2]
  return -(x0**2 + 0.1*(x1 - 1)**2 + 0.5*(x2 + 2)**2) # maximum reward is zero


if __name__ == '__main__':

  optimal_params = [0, 1, -2] # these are the optimal weights with respect to the reward function

  fig, axs = plt.subplots(2, 3, sharex=True, sharey=False)

  for i in range(3):
    population_size=2000
    sigma=10.0**(-i-1)
    lr=1e-3
    initial_params=np.random.randn(3)
    num_iters=500

    best_params, rewards = evolution_strategy(
      f=reward_function,
      population_size=population_size,
      sigma=sigma,
      lr=lr,
      initial_params=initial_params,
      num_iters=num_iters,
    )

    new_best_params, new_rewards = newtons_method_evolution_strategy(
      f=reward_function,
      population_size=population_size,
      sigma=sigma,
      lr=lr,
      initial_params=initial_params,
      num_iters=num_iters,
    )

    # final params
    print("Final params error:", np.sum(np.abs(best_params - optimal_params)))
    print("Final Newton's params error:", np.sum(np.abs(new_best_params - optimal_params)))

    # plot the rewards per iteration
    axs[0, i].plot(rewards, color='b')
    axs[0, i].plot(new_rewards, color='g')
    axs[0, i].set_title('sigma='+str(sigma))
    axs[0, i].legend(['Original', 'Newtons'])
    axs[0, i].set_xlabel('Iterations')
    axs[0, i].set_ylabel('Return')

    axs[1, i].plot(rewards, color='b')
    axs[1, i].plot(new_rewards, color='g')
    axs[1, i].set(ylim=(-0.05,0.05))
    axs[1, i].set_title('sigma='+str(sigma))
    axs[1, i].legend(['Original', 'Newtons'])
    axs[1, i].set_xlabel('Iterations')
    axs[1, i].set_ylabel('Return')


  plt.tight_layout()
  plt.show()

