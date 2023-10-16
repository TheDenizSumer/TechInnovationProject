import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab 

import matplotlib
x = np.arange(1, 10)
y = x.reshape(-1, 1)
h = x * y

cs = plt.contourf(h, levels=[10, 30, 50],
    colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
cs.cmap.set_over('red')
cs.cmap.set_under('blue')
cs.changed()

plt.figure()
plt.clabel(cs, fontsize=10)

plt.title('Simplest default with labels')

plt.show()