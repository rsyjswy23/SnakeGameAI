import matplotlib.pyplot as plt
from IPython import display
from matplotlib.figure import Figure
from mpl_toolkits.axisartist.axislines import Subplot 
from matplotlib.ticker import MaxNLocator


plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.figure(facecolor = 'black', frameon = True))

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.tick_params(axis='x', labelcolor='white')
    ax.tick_params(axis='y', labelcolor='white')
    ax.set_facecolor('black')

    plt.title('Process of AI Learns to Play Snake', color='white')
    plt.xlabel('Number of Games', color='white')
    plt.ylabel('Score', color='white')

    plt.plot(scores, label = 'Score', linewidth=2, color='cornflowerblue')
    plt.plot(mean_scores, label = 'Mean Score', linewidth=2, color='salmon')
   # plt.xlim((1, 200))
    plt.ylim((0, 60))
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.yaxis.get_major_locator().set_params(integer=True)
   # plt.text(len(scores)-1, scores[-1], str(scores[-1]), color='cornflowerblue')
   # plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]), color='salmon')
    plt.show(block = False)
    #legend = plt.legend(loc='upper right', facecolor='black', framealpha=1)
    #legend.get_frame().set_facecolor('C0')
    ax.legend(framealpha=1)
    plt.show()
    plt.pause(.1)
    