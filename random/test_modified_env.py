from modified_env import *

def step_test():
    aug_lander = IntLunarLander()
    obs, info = aug_lander.reset(seed=42)
    MAX_STEPS=100_000
    for _ in range(MAX_STEPS):
        s, ext_r, int_r, term, _, _ = aug_lander.extint_step(1)
        print(f'State: {s}, External reward: {ext_r} Internal reward: {int_r}')
        if term:
            break
    
    
if __name__ == '__main__':
    step_test()