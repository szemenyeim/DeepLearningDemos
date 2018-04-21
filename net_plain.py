import torch
from torch.autograd import Variable
import visualize

torch.manual_seed(1)

N,D_in,H,D_out = 64,1000,100,10

lrs = [2.5e-6, 2e-6, 1e-6, 7.5e-7, 5e-7]

plotter = visualize.LinePlotter("CVSDemo")

x = Variable(torch.randn(N, D_in), requires_grad=False)
y = Variable(torch.randn(N, D_out), requires_grad=False)

for i in range(5):

    w1 = Variable(torch.randn(D_in,H),requires_grad = True)
    w2 = Variable(torch.randn(H,D_out),requires_grad = True)

    learning_rate = lrs[i]

    for t in range(100):

        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        loss = (y_pred-y).pow(2).sum()

        if w1.grad is not None: w1.grad.data.zero_()
        if w2.grad is not None: w2.grad.data.zero_()

        loss.backward()

        w1.data -= learning_rate*w1.grad.data
        w2.data -= learning_rate*w2.grad.data

        err = loss.data[0]

        plotter.plot("LR=%.2E" % lrs[i], "Error", t,err)

#vis.line(err)
#print err
