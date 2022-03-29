from pettingzoo.mpe import simple_v2
from pettingzoo.mpe.SymmetricGame import simple_sg
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter('./log')

env=simple_sg.env(max_cycles=10*60,continuous_actions=True)
env.reset()
a0=env.agents[0]
a1=env.agents[-1]
for i in range(10*60):
    if i%10 == 0:
        print("i!")
    # env.render()
    env.step(np.random.rand(5).astype(np.float32))
    env.step(np.random.rand(5).astype(np.float32))
    ob1=env.observe(a0)
    ob2=env.observe(a0)
    r0=env.rewards[a0]
    r1=env.rewards[a1]
    writer.add_scalar('reward/reward0',r0,i)
    writer.add_scalar('reward/reward1',r1,i)
    

