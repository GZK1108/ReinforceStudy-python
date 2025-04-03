from utils import Environment, Policy
from BOE import policy_iteration, value_iteration, bellman_equation

env = Environment(5, 5)
env.reset()
policy = Policy(env)
value_iteration(env, policy)
V = bellman_equation(env, policy)
print(V)

