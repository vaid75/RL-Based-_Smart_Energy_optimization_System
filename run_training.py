import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if os.path.exists('dqn_model.pth'):
    os.remove('dqn_model.pth')
    print('Removed old model')

from HEMS import HEMS
import numpy as np

h = HEMS()
rewards, savings = h.train(n_episodes=300, epsilon_decay=0.98, steps=1000)

print()
print('=== TRAINING SUMMARY ===')
print('Ep 1-10    avg reward  : {:.4f}'.format(np.mean(rewards[:10])))
print('Ep 101-110 avg reward  : {:.4f}'.format(np.mean(rewards[100:110])))
print('Ep 291-300 avg reward  : {:.4f}'.format(np.mean(rewards[290:])))
print('Ep 1-10    avg savings : {:.6f}'.format(np.mean(savings[:10])))
print('Ep 291-300 avg savings : {:.6f}'.format(np.mean(savings[290:])))
print('Positive savings eps   : {} / {}'.format(sum(1 for s in savings if s > 0), len(savings)))

print()
print('Running test episode...')
result = h.test(steps=200)
baseline = result['baseline_cost']
cost     = result['cost']
pct      = (baseline - cost) / baseline * 100 if baseline > 0 else 0
print('Test  baseline cost : {:.4f}'.format(baseline))
print('Test  actual cost   : {:.4f}'.format(cost))
print('Test  savings pct   : {:.2f}%'.format(pct))
bat_key = 'battery'
print('Test  battery final : {}%'.format(result[bat_key]))
