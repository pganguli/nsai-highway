import os
import sys

from config import BASE_ENV_CONFIG, DQN_KWARGS
from environments import make_env
from stable_baselines3 import DQN

sys.path.insert(0, ".")
os.makedirs("models/neural", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 500-step test for neural
env = make_env(BASE_ENV_CONFIG)
kw = {**DQN_KWARGS, "verbose": 0}
model = DQN(env=env, **kw, seed=0)
model.learn(500, progress_bar=False)
model.save("models/neural/model_test")
print("Neural 500-step test: OK")

# 500-step test for shielded
senv = make_env(BASE_ENV_CONFIG, shielded=True)
model2 = DQN(env=senv, **kw, seed=0)
model2.learn(500, progress_bar=False)
model2.save("models/neurosymbolic/model_test")
print(f"Shielded 500-step test: OK  override_rate={senv.shield.override_rate:.2%}")  # type: ignore
env.close()
senv.close()
