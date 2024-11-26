import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

class Window:
    def __init__(self, population, weights_list, baseline):
        self.population = population
        self.weights_list = weights_list
        self.baseline = baseline
        eps_init = 0.1
        self.epsilon = eps_init
        self.ngen = 1000
        self.N = len(self.weights_list)
        self.fig = plt.figure(figsize=(15, 9))
        self.Rav_trace_coll = [[]]
        self.Rk_trace_list = [[] for _ in range(self.N)]
        self.Rkbar_list = [self.baseline for _ in range(self.N)]
        self.nk_list = [0 for _ in range(self.N)]
        # params
        self.rho = 0.1
        # main plots and buttons
        self.Rav_ax = self.fig.add_axes([0.2, 0.69, 0.79, 0.30])
        self.Rkbar_ax = self.fig.add_axes([0.2, 0.36, 0.79, 0.30])
        self.nk_ax = self.fig.add_axes([0.2, 0.03, 0.79, 0.30])
        indices = [k + 1 for k in range(self.N)]
        self.Rkbar_plot = self.Rkbar_ax.bar(indices, self.Rkbar_list)
        self.nk_plot = self.nk_ax.bar(indices, self.nk_list)
        self.Rav_plot_coll = []
        Rav_plot, = self.Rav_ax.plot([])
        self.Rav_plot_coll.append(Rav_plot)
        fr_left = 0.025
        fr_bot = 0.02
        fr_top = 0.98
        height = 0.1
        sep = 0.05
        width = 0.1
        new_ax = self.fig.add_axes([fr_left, fr_top - height, width, height])
        self.new_button = Button(new_ax, 'New trial')
        self.new_button.on_clicked(self.new_action)
        gen_ax = self.fig.add_axes([fr_left, fr_bot, width, height])
        self.gen_button = Button(gen_ax, f'{self.ngen} steps')
        self.gen_button.on_clicked(self.gen_action)
        self.Bk_list = []
        for k in range(self.N):
            top = fr_top - (k + 1) * (height + sep) - height
            Bk_ax = self.fig.add_axes([fr_left, top, width, height])
            self.Bk_list.append(Button(Bk_ax, f'{k + 1}'))
            Bk_on_click = lambda event, k=k : self.Bk_action(k, event)
            self.Bk_list[k].on_clicked(Bk_on_click)
        eps_ax = self.fig.add_axes([fr_left, fr_bot + height + sep, width, height])
        self.eps_slider = Slider(eps_ax, label='epsilon',
                                 valmin=0.0, valmax=0.2, valinit=eps_init)
        self.eps_slider.on_changed(self.eps_action)

    def find_lims(self, values, vmin, vmax):
            vmin = min(min(values), vmin)
            vmax = max(max(values), vmax)
            center = 0.5 * (vmax + vmin)
            width = vmax - vmin
            radius = (1 + self.rho) * width / 2
            return center - radius, center + radius        

    def draw_plot(self):
        self.Rkbar_ax.set_ylim(self.find_lims(self.Rkbar_list, 0, 0))
        self.nk_ax.set_ylim(self.find_lims(self.nk_list, 0, 0))
        Rav_xmin, Rav_xmax = 0, 0
        Rav_ymin, Rav_ymax = 0, 0
        for Rav_trace in self.Rav_trace_coll:
             if Rav_trace:
                xrange = range(len(Rav_trace))
                Rav_xmin_cur, Rav_xmax_cur = self.find_lims(xrange, 0, 0)
                Rav_xmin = min(Rav_xmin, Rav_xmin_cur)
                Rav_xmax = max(Rav_xmax, Rav_xmax_cur)
                Rav_ymin_cur, Rav_ymax_cur = self.find_lims(Rav_trace, 0, 0)
                Rav_ymin = min(Rav_ymin, Rav_ymin_cur)
                Rav_ymax = max(Rav_ymax, Rav_ymax_cur)
        self.Rav_ax.set_xlim(Rav_ymin, Rav_xmax)
        self.Rav_ax.set_ylim(Rav_ymin, Rav_ymax)
        plt.draw()

    def new_action(self, event):
        self.Rk_trace_list = [[] for _ in range(self.N)]
        self.Rkbar_list = [self.baseline for _ in range(self.N)]
        self.nk_list = [0 for _ in range(self.N)]
        self.Rav_trace_coll.append([])
        for (k, Rkbar) in enumerate(self.Rkbar_list):
            self.Rkbar_plot[k].set_height(Rkbar)
        for (k, nk) in enumerate(self.nk_list):
            self.nk_plot[k].set_height(nk)
        Rav_plot, = self.Rav_ax.plot([])
        self.Rav_plot_coll.append(Rav_plot)
        self.draw_plot()

    def Bk_action(self, k, event):
        R = random.choices(self.population, self.weights_list[k])[0]
        self.Rk_trace_list[k].append(R)
        nk = len(self.Rk_trace_list[k])
        self.nk_list[k] = nk
        Rkbar = sum(self.Rk_trace_list[k], self.baseline) / nk
        self.Rkbar_list[k] = Rkbar
        self.Rkbar_plot[k].set_height(Rkbar)
        self.nk_plot[k].set_height(nk)
        tot_R = 0.0
        nstep = 0
        for Rk_trace in self.Rk_trace_list:
            nstep += len(Rk_trace)
            tot_R += sum(Rk_trace, 0.0)
        self.Rav_trace_coll[-1].append(tot_R / nstep)
        self.Rav_plot_coll[-1].set_xdata(range(nstep))
        self.Rav_plot_coll[-1].set_ydata(self.Rav_trace_coll[-1])
        self.draw_plot()

    def find_max(self, list):
        i_opt = -1
        value_opt = -float('inf')
        for (i, value) in enumerate(list):
            if value > value_opt:
                value_opt = value
                i_opt = i
        return i_opt

    def gen_action(self, event):
        for _ in range(self.ngen):
            k = self.find_max(self.Rkbar_list)
            r = random.random()
            if r < self.epsilon:
                k = random.choices(range(self.N))[0]
            R = random.choices(self.population, self.weights_list[k])[0]
            self.Rk_trace_list[k].append(R)
            nk = len(self.Rk_trace_list[k])
            self.nk_list[k] = nk
            Rkbar = sum(self.Rk_trace_list[k], self.baseline) / nk
            self.Rkbar_list[k] = Rkbar
            tot_R = 0.0
            nstep = 0
            for Rk_trace in self.Rk_trace_list:
                nstep += len(Rk_trace)
                tot_R += sum(Rk_trace, 0.0)
            self.Rav_trace_coll[-1].append(tot_R / nstep)
        for (k, Rkbar) in enumerate(self.Rkbar_list):
            self.Rkbar_plot[k].set_height(Rkbar)
        for (k, nk) in enumerate(self.nk_list):
            self.nk_plot[k].set_height(nk)
        self.Rav_plot_coll[-1].set_xdata(range(len(self.Rav_trace_coll[-1])))
        self.Rav_plot_coll[-1].set_ydata(self.Rav_trace_coll[-1])
        self.draw_plot()

    def eps_action(self, val):
        self.epsilon = val
        print(f'eps: {self.epsilon}')