import numpy as np
from tick.plot import plot_point_process
from tick.hawkes import SimuHawkes, HawkesExpKern, HawkesKernelExp
import csv
import math
import random

with open('Book3.csv') as f:
    reader = csv.reader(f, delimiter=';')
    reader.__next__()
    data_1dim = []
    data_2dim = []
    co = 1
    for i in reader:
        if float(i[0]) < -0.025:
            data_1dim.append(float(co))
        if float(i[1]) < -0.025:
            data_2dim.append(float(co))
        co += 1
    data = [np.array(np.array(data_1dim)), np.array(np.array(data_2dim))]

with open('Book3.csv') as f:
    reader = csv.reader(f, delimiter=';')
    reader.__next__()
    test_1dim = []
    test_2dim = []
    co = 1
    for i in reader:
        if float(i[0]) < -0.025:
            test_1dim.append(float(co))
        if float(i[1]) < -0.025:
            test_2dim.append(float(co))
        co += 1
    test = [np.array(np.array(data_1dim)), np.array(np.array(data_2dim))]

print(data)
co = 0
re = []
for i1 in range(1, 10):
    for i2 in range(1, 10):
        for i3 in range(1, 10):
            for i4 in range(1, 10):
                dec = [[float(i1 / 100) + 0.05, float(i2 / 100) + 0.1],
                       [float(i3 / 100) + 0.05, float(i4 / 100) + 0.1]]
                learner = HawkesExpKern(dec)
                learner.fit(data)
                if learner.score() > co:
                    co = learner.score()
                    re = dec
print(re)
learner = HawkesExpKern(re)
learner.fit(data)
print(learner.adjacency, learner.baseline, co)

co_true1 = 0
co_true2 = 0
co_false1 = 0
co_false2 = 0
for _ in range(1, 101):
    for i in range(20, 749):
        fl = 0
        intens = [learner.baseline[1]] * 5
        for j in range(-20, 0):
            if float(i + j) in test[0]:
                fl = 1
                for k in range(5):
                    intens[k] += learner.adjacency[1][0] * math.exp(re[1][0] * (j - k))
            if float(i + j) in test[1]:
                fl = 1
                for k in range(5):
                    intens[k] += learner.adjacency[1][1] * math.exp(re[1][1] * (j - k))
        if not fl:
            if (float(i) in test[1] or float(i) in test[1] or float(i) in test[1] or float(i) in test[1] or
                    float(i) in test[1]):
                co_false2 += 1
            else:
                co_true2 += 1
            continue
        co = 0
        for k in range(5):
            base = intens[k]
            kernel11 = HawkesKernelExp(0.0, 1.0)
            hawkes = SimuHawkes(kernels=[[kernel11]],
                                end_time=1, verbose=False, baseline=[base])
            hawkes.simulate()
            if hawkes.n_total_jumps > 0:
                co = 1
                break
        if co:
            if (float(i) in test[1] or float(i) in test[1] or float(i) in test[1] or float(i) in test[1] or
                    float(i) in test[1]):
                co_true1 += 1
            else:
                co_false1 += 1
        else:
            if (float(i) in test[1] or float(i) in test[1] or float(i) in test[1] or float(i) in test[1] or
                    float(i) in test[1]):
                co_false2 += 1
            else:
                co_true2 += 1
print(co_true1 / 100, co_true2 / 100, co_false1 / 100, co_false2 / 100)

run_time = 1000

kernel11 = HawkesKernelExp(learner.adjacency[0][0], re[0][0])
kernel12 = HawkesKernelExp(learner.adjacency[0][1], re[0][1])
kernel21 = HawkesKernelExp(learner.adjacency[1][0], re[1][0])
kernel22 = HawkesKernelExp(learner.adjacency[1][1], re[1][1])
hawkes = SimuHawkes(kernels=[[kernel11, kernel12],
                             [kernel21, kernel22]],
                    end_time=run_time, verbose=False, baseline=learner.baseline)


dt = 0.1

hawkes.track_intensity(dt)
hawkes.simulate()
plot_point_process(hawkes)

'''import matplotlib.pyplot as plt

from tick.plot import plot_hawkes_kernels
from tick.hawkes import SimuHawkesSumExpKernels, SimuHawkesMulti, \
    HawkesSumExpKern

end_time = 1000
n_realizations = 10

decays = [.5, 2., 6.]
baseline = [0.12, 0.07]
adjacency = [[[0, .1, .4], [.2, 0., .2]],
             [[0, 0, 0], [.6, .3, 0]]]

hawkes_exp_kernels = SimuHawkesSumExpKernels(
    adjacency=adjacency, decays=decays, baseline=baseline,
    end_time=end_time, verbose=False, seed=1039)

hawkes_exp_kernels.end_time = 1000
hawkes_exp_kernels.simulate()


learner = HawkesSumExpKern(decays, penalty='elasticnet',
                           elastic_net_ratio=0.8)
learner.fit(hawkes_exp_kernels.timestamps)

fig = plot_hawkes_kernels(learner, hawkes=hawkes_exp_kernels, show=False)

for ax in fig.axes:
    ax.set_ylim([0., 1.])'''