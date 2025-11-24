from simglucose.envs import T1DSimGymnaisumEnv
from simglucose_env_wrapper import SimglucoseWrapper

# Create and wrap environment
base_env = T1DSimGymnaisumEnv(patient_name="adolescent#001")
wrapped_env = SimglucoseWrapper(base_env)

# Test reset
state, info = wrapped_env.reset()
print(f"Initial state shape: {state.shape}")
print(f"Initial state: {state}")
print(f"Observation space: {wrapped_env.observation_space}")
print(f"Action space: {wrapped_env.action_space}")

# Test a few steps
for i in range(100):
    # Random action
    action = wrapped_env.action_space.sample()
    print(f"\nAction: ${action}")
    state, reward, done, truncated, info = wrapped_env.step(action)
    
    print(f"\nStep {i+1}:")
    print(f"  CGM: {state[0]:.1f}")
    print(f"  Trend: {state[1]:.2f}")
    print(f"  IOB: {state[2]:.2f}")
    print(f"  Meal: {state[5]:.2f} U")
    print(f"  Reward: {reward:.2f}")
    
    if done:
        print("Episode terminated!")
        break