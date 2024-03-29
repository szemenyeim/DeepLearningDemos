import visdom
import numpy as np

class LinePlotter(object):
    def __init__(self, env_name="main"):
        self.vis = visdom.Visdom(port=8097)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(X=np.array([x, x]),
                                        Y=np.array([y, y]), env=self.env, update=None, opts=dict(
                                        legend=[split_name],
                                        title=var_name,
                                        xlabel="Iters",
                                        ylabel=var_name
                                        ))
        else:
            self.vis.line(X=np.array([x, x]), Y=np.array([y, y]),  env=self.env, update='append',
                                win=self.plots[var_name], name=split_name)
