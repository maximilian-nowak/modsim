import numpy as np
import matplotlib.pyplot as plt

def m1(r): return 0.2 + 0.001*r
def m2(r): return 1.0 + 0.0005*r
def m3(r): return 0.5 + 0.002*r

x = np.linspace(0, 24000, 24000)

plt.xlim(0,24000)
plt.plot(x, m1(x))
plt.plot(x, m2(x))
plt.plot(x, m3(x))
plt.title('Messgeräte')
plt.ylabel('sigma des Messfehlers')
plt.xlabel('mm')
plt.legend(['m1', 'm2', 'm3'])
plt.show()