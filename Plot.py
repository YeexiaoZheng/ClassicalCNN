import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

class myplt:
    def __init__(self):
        pass

    def line3dim(model_type, xlabel, ylabel, zlabel, xlist, ylist, zlist):
        plt.figure(figsize=(15, 6))
        plt.title(model_type + ' ' + ylabel + ' with ' + xlabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(
            ticks=range(0, len(ylist[0]), int(len(ylist[0]) / len(xlist))), 
            labels=xlist,
        )

        for i in range(len(zlist)):
            plt.plot(ylist[i], label=zlist[i])
            
        plt.legend()
        plt.savefig(model_type + '_' + ylabel + '.png')

    def bar(model_type, xlabel, ylabel, xlist, ylist):
        plt.figure()
        plt.title(model_type + ' ' + ylabel + ' with ' + xlabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(range(len(xlist)), xlist, rotation=45)

        plt.bar(range(len(xlist)), ylist, color=plt.get_cmap('Set3')(range(len(xlist))))
        for x, y in zip(range(len(xlist)), ylist):
            plt.annotate(y, (x, y), ha='center', va='bottom')

        plt.savefig(model_type + '_' + ylabel + '.png', bbox_inches = 'tight')