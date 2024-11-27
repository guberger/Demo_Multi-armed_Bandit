import math
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

class Window:
    def __init__(self, population, weights_list):
        self.population = population
        self.weights_list = weights_list
        self.baseline = 0
        self.epsilon = 0.1
        self.ngen = 1000
        self.N = len(self.weights_list)
        self.fig = plt.figure(figsize=(15, 9))
        self.Rav_trace = []
        self.Rk_trace_list = [[] for _ in range(self.N)]
        self.Rkbar_list = [self.baseline for _ in range(self.N)]
        self.nk_list = [0 for _ in range(self.N)]
        # params
        self.rho = 0.1
        # main plots and buttons
        self.Rav_ax = self.fig.add_axes([0.2, 0.69, 0.79, 0.30])
        self.Rkbar_ax = self.fig.add_axes([0.2, 0.36, 0.79, 0.30])
        self.nk_ax = self.fig.add_axes([0.2, 0.03, 0.79, 0.30])
        self.Rav_ax.set_ylabel('Average reward')
        self.Rkbar_ax.set_ylabel('Estimated expected reward per bandit')
        self.nk_ax.set_ylabel('Number of visit per action')
        self.Rav_xlim = [0, 0]
        self.Rav_ylim = [0, 0]
        self.Rkbar_ylim = [0, 0]
        self.nk_ylim = [0, 0]
        indices = [k + 1 for k in range(self.N)]
        self.Rkbar_plot = self.Rkbar_ax.bar(indices, self.Rkbar_list)
        self.nk_plot = self.nk_ax.bar(indices, self.nk_list)
        self.Rav_plot_coll = [self.Rav_ax.plot([])[0]]
        fr_left = 0.035
        fr_bot = 0.02
        fr_top = 0.98
        height = 0.05
        sep = 0.02
        width = 0.07
        new_ax = self.fig.add_axes([fr_left, fr_top - height, width, height])
        self.new_button = Button(new_ax, 'New trial')
        self.new_button.on_clicked(self.new_action)
        self.Bk_list = []
        for k in range(self.N):
            top = fr_top - (k + 1) * (height + sep) - height
            Bk_ax = self.fig.add_axes([fr_left, top, width, height])
            self.Bk_list.append(Button(Bk_ax, f'{k + 1}'))
            Bk_on_click = lambda event, k=k : self.Bk_action(k, event)
            self.Bk_list[k].on_clicked(Bk_on_click)
        ucb_ax = self.fig.add_axes([fr_left, fr_bot, width, height])
        self.ucb_button = Button(ucb_ax, f'{self.ngen} steps UCB')
        self.ucb_button.on_clicked(self.ucb_action)
        greedy_ax = self.fig.add_axes(
            [fr_left, fr_bot + height + sep, width, height]
        )
        self.greedy_button = Button(greedy_ax, f'{self.ngen} steps greedy')
        self.greedy_button.on_clicked(self.greedy_action)
        eps_ax = self.fig.add_axes(
            [fr_left, fr_bot + (height + sep) * 2, width, height]
        )
        self.eps_slider = Slider(eps_ax, label='eps',
                                 valmin=0.0, valmax=0.2,
                                 valinit=self.epsilon)
        self.eps_slider.on_changed(self.eps_action)
        base_ax = self.fig.add_axes(
            [fr_left, fr_bot + (height + sep) * 3, width, height]
        )
        self.base_slider = Slider(base_ax, label='base',
                                  valmin=0, valmax=max(self.population),
                                  valinit=self.baseline)
        self.base_slider.on_changed(self.base_action)
        self.text_ax = self.fig.add_axes(
            [fr_left, fr_bot + (height + sep) * 4, width, height]
        )
        self.text_ax.set_xticks([])
        self.text_ax.set_yticks([])
        self.text_box = self.text_ax.text(0.5, 0.5, '',
                                          horizontalalignment='center')

    def find_lims(self, values, vmin, vmax):
            vmin = min(min(values), vmin)
            vmax = max(max(values), vmax)
            center = 0.5 * (vmax + vmin)
            width = vmax - vmin
            radius = (1 + self.rho) * width / 2
            return center - radius, center + radius        

    def draw_plot(self):
        self.Rkbar_ax.set_ylim(self.find_lims(self.Rkbar_ylim, 0, 0))
        self.nk_ax.set_ylim(self.find_lims(self.nk_ylim, 0, 0))
        self.Rav_ax.set_xlim(self.find_lims(self.Rav_xlim, 0, 0))
        self.Rav_ax.set_ylim(self.find_lims(self.Rav_ylim, 0, 0))
        plt.draw()

    def new_action(self, event):
        self.Rav_trace = []
        self.Rk_trace_list = [[] for _ in range(self.N)]
        self.Rkbar_list = [self.baseline for _ in range(self.N)]
        self.nk_list = [0 for _ in range(self.N)]
        self.Rkbar_ylim = [0, 0]
        self.nk_ylim = [0, 0]
        for (k, Rkbar) in enumerate(self.Rkbar_list):
            self.Rkbar_plot[k].set_height(Rkbar)
        for (k, nk) in enumerate(self.nk_list):
            self.nk_plot[k].set_height(nk)
        self.Rav_plot_coll.append(self.Rav_ax.plot([])[0])
        self.draw_plot()

    def Bk_action(self, k, event):
        R = random.choices(self.population, self.weights_list[k])[0]
        self.text_box.set_text(f'{R}')
        self.Rk_trace_list[k].append(R)
        nk = len(self.Rk_trace_list[k])
        self.nk_list[k] = nk
        Rkbar = sum(self.Rk_trace_list[k], self.baseline) / nk
        self.Rkbar_list[k] = Rkbar
        #
        self.Rkbar_ylim[1] = max(self.Rkbar_ylim[1], Rkbar)
        self.nk_ylim[1] = max(self.nk_ylim[1], nk)
        self.Rkbar_plot[k].set_height(Rkbar)
        self.nk_plot[k].set_height(nk)
        #
        tot_R = 0.0
        nstep = 0
        for Rk_trace in self.Rk_trace_list:
            nstep += len(Rk_trace)
            tot_R += sum(Rk_trace, 0.0)
        Rav = tot_R / nstep
        self.Rav_trace.append(Rav)
        assert nstep == len(self.Rav_trace)
        #
        self.Rav_xlim[1] = max(self.Rav_xlim[1], nstep)
        self.Rav_ylim[0] = min(self.Rav_ylim[0], Rav)
        self.Rav_ylim[1] = max(self.Rav_ylim[1], Rav)
        self.Rav_plot_coll[-1].set_xdata(range(nstep))
        self.Rav_plot_coll[-1].set_ydata(self.Rav_trace)
        self.draw_plot()

    def find_max(self, list):
        i_opt = -1
        value_opt = -float('inf')
        for (i, value) in enumerate(list):
            if value > value_opt:
                value_opt = value
                i_opt = i
        return i_opt
    
    def ucb_action(self, event):
        c = max(self.population) - min(self.population)
        for _ in range(self.ngen):
            S = c * math.log(max(len(self.Rav_trace), 1))
            UCB_list = [
                self.Rkbar_list[k] +
                math.sqrt(S / max(self.nk_list[k], 1e-9))
                for k in range(self.N)
            ]
            k = self.find_max(UCB_list)
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
            Rav = tot_R / nstep
            self.Rav_trace.append(Rav)
            assert nstep == len(self.Rav_trace)
        for (k, Rkbar) in enumerate(self.Rkbar_list):
            self.Rkbar_ylim[1] = max(self.Rkbar_ylim[1], Rkbar)
            self.Rkbar_plot[k].set_height(Rkbar)
        for (k, nk) in enumerate(self.nk_list):
            self.nk_ylim[1] = max(self.nk_ylim[1], nk)
            self.nk_plot[k].set_height(nk)
        #
        nstep = len(self.Rav_trace)
        self.Rav_xlim[1] = max(self.Rav_xlim[1], nstep)
        self.Rav_ylim[0] = min(self.Rav_ylim[0], min(self.Rav_trace))
        self.Rav_ylim[1] = max(self.Rav_ylim[1], max(self.Rav_trace))
        self.Rav_plot_coll[-1].set_xdata(range(nstep))
        self.Rav_plot_coll[-1].set_ydata(self.Rav_trace)
        self.draw_plot()

    def greedy_action(self, event):
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
            Rav = tot_R / nstep
            self.Rav_trace.append(Rav)
            assert nstep == len(self.Rav_trace)
        for (k, Rkbar) in enumerate(self.Rkbar_list):
            self.Rkbar_ylim[1] = max(self.Rkbar_ylim[1], Rkbar)
            self.Rkbar_plot[k].set_height(Rkbar)
        for (k, nk) in enumerate(self.nk_list):
            self.nk_ylim[1] = max(self.nk_ylim[1], nk)
            self.nk_plot[k].set_height(nk)
        #
        nstep = len(self.Rav_trace)
        self.Rav_xlim[1] = max(self.Rav_xlim[1], nstep)
        self.Rav_ylim[0] = min(self.Rav_ylim[0], min(self.Rav_trace))
        self.Rav_ylim[1] = max(self.Rav_ylim[1], max(self.Rav_trace))
        self.Rav_plot_coll[-1].set_xdata(range(nstep))
        self.Rav_plot_coll[-1].set_ydata(self.Rav_trace)
        self.draw_plot()

    def eps_action(self, val):
        self.epsilon = val
        print(f'eps: {self.epsilon}')

    def base_action(self, val):
        self.baseline = val
        print(f'base: {self.baseline}')
        for (k, Rk_trace) in enumerate(self.Rk_trace_list):
            nk = max(len(Rk_trace), 1)
            Rkbar = sum(Rk_trace, self.baseline) / nk
            self.Rkbar_list[k] = Rkbar
            self.Rkbar_plot[k].set_height(Rkbar)
        self.draw_plot()