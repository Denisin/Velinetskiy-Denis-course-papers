import numpy as np
from tick.plot import plot_point_process, plot_hawkes_kernels
from tick.hawkes import SimuHawkes, HawkesKernelSumExp, HawkesExpKern, HawkesKernelExp
import matplotlib.pyplot as plt
import csv
import math
import random

with open('Book1_1.csv') as f:
    reader = csv.reader(f, delimiter=';')
    reader.__next__()
    data = []
    co = 1
    for i in reader:
        if float(i[1]) < -0.025:
            data.append(float(co))
        co += 1
    data = [np.array(np.array(data))]

with open('test2.csv') as f:
    reader = csv.reader(f, delimiter=';')
    reader.__next__()
    test = []
    co = 1
    for i in reader:
        if float(i[1]) < -0.025:
            test.append(float(co))
        co += 1
    test = [np.array(np.array(data))]

print(data)

co = 0
adj = 0
base = 0
dec = 0
for i in range(1, 2000):
    learner = HawkesExpKern(float(i / 1000))
    learner.fit(data)
    if learner.score() > co:
        adj = learner.adjacency[0][0]
        base = learner.baseline[0]
        dec = learner.decays
        co = learner.score()
print(adj * dec, base, dec)

'''run_time = 1000

hawkes = SimuHawkes(n_nodes=1, end_time=run_time, verbose=False)
kernel = HawkesKernelExp(adj, dec)
hawkes.set_kernel(0, 0, kernel)
hawkes.set_baseline(0, base)

dt = 0.1
hawkes.track_intensity(dt)
hawkes.simulate()
timestamps = hawkes.timestamps
print(timestamps)

plot_point_process(hawkes, n_points=50000, t_min=2, t_max=1000, show=True)
plt.show()
print(data)'''

co_true1 = 0
co_true2 = 0
co_false1 = 0
co_false2 = 0
for _ in range(100):

    for i in range(20, 749):
        fl = 0
        intens = [base] * 5
        for j in range(-20, 0):
            if float(i + j) in test[0]:
                fl = 1
                for k in range(5):
                    intens[k] += adj * math.exp(dec * (j - k))
        if not fl:
            if (float(i) in test[0] or float(i) in test[0] or float(i) in test[0] or float(i) in test[0] or
                    float(i) in test[0]):
                co_false2 += 1
            else:
                co_true2 += 1
            continue
        co = 0
        for k in range(5):
            hawkes = SimuHawkes(n_nodes=1, end_time=1, verbose=False)
            kernel = HawkesKernelExp(adj, dec)
            hawkes.set_kernel(0, 0, kernel)
            hawkes.set_baseline(0, intens[k])
            hawkes.simulate()
            if hawkes.n_total_jumps > 0:
                co = 1
                break
        if co:
            if (float(i) in test[0] or float(i) in test[0] or float(i) in test[0] or float(i) in test[0] or
                    float(i) in test[0]):
                co_true1 += 1
            else:
                co_false1 += 1
        else:
            if (float(i) in test[0] or float(i) in test[0] or float(i) in test[0] or float(i) in test[0] or
                    float(i) in test[0]):
                co_false2 += 1
            else:
                co_true2 += 1
print(co_true1 / 100, co_true2 / 100, co_false1 / 100, co_false2 / 100)
