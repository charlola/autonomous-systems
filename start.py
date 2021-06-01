import mlagents
from mlagents_envs.environment import UnityEnvironment
env = UnityEnvironment(file_name="env/UnityEnvironment")

env.reset()

print("Behavior Names", env.get_behavior_names())

behavior_name = list(env.get_behavior_names())[0]
print(f"Name of the behavior : {behavior_name}")
spec = env.get_behavior_spec(behavior_name)


print("Number of observations : ", len(spec.observation_shapes))

if spec.is_action_continuous():
  print("The action is continuous")

if spec.is_action_discrete():
  print("The action is discrete")


print(list(filter(lambda x: not x.startswith("_"), dir(spec))))

decision_steps, terminal_steps = env.get_steps(behavior_name)
print(decision_steps.obs)

for episode in range(3):
  env.reset()
  decision_steps, terminal_steps = env.get_steps(behavior_name)
  # -1 indicates not yet tracking
  tracked_agent = -1 
  # For the tracked_agent
  done = False 
  # For the tracked_agent
  episode_rewards = 0 
  while not done:
    # Track the first agent we see if not tracking
    # Note : len(decision_steps) = [number of agents that requested a decision]
    if tracked_agent == -1 and len(decision_steps) >= 1:
      tracked_agent = decision_steps.agent_id[0]
    # Generate an action for all agents
    action = spec.create_empty_action(len(decision_steps))
    # Set the actions
    env.set_actions(behavior_name, action)
    # Move the simulation forward
    env.step()
    # Get the new simulation results
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    # The agent requested a decision
    if tracked_agent in decision_steps:
      episode_rewards += decision_steps[tracked_agent].reward
      # The agent terminated its episode
    if tracked_agent in terminal_steps:
      episode_rewards += terminal_steps[tracked_agent].reward
      done = True
  print(f"Total rewards for episode {episode} is {episode_rewards}")



env.close()
print("Closed environment")