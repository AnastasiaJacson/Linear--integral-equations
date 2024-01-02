import numpy as np
from tqdm import tqdm
from utils import *
import pandas as pd

ns = [2, 4, 8, 16, 32, 64]
results = []

x = np.array([.4, .7])
m_phi, m_u = 40, 40

exact = lambda x: x[0] * x[1]
f = lambda t: exact(xi(t))

for n in tqdm(ns):
    u, phi = find_u(n=n, m_phi=m_phi, m_u=m_u, f=f)
    results.append({'n': n, 'approx': u(x), 'exact': exact(x),'error': abs(exact(x) - u(x))})

df = pd.DataFrame(results)

print(df)

plt.plot(df.n, df.error)
plt.ylim(0, .1)

plt.show()

plt.plot(df.n, df.approx, label='approx')
plt.plot(df.n, df.exact, label='exact')

plt.ylim(1.5, 2.5)

plt.legend()

plt.show()
