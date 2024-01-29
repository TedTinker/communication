#%%

from utils import args
from agent import Agent

args.alpha = .1
args.delta = 1

agent = Agent(1, GUI = True, args = args)
agent.training()
# %%
