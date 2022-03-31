import matplotlib.pyplot as plt

#Arithmetic mean
class CreateGraph:
    def __init__(self, batch_count,name):

        self.batch_count = batch_count
        self.num_for_avg = 0
        self.array_epoch, self.array_loss =  [],[]
        self.name = name

    def count(self,epoch):

        self.array_epoch.append(epoch)
        self.array_loss.append(self.num_for_avg/ self.batch_count)
        #print(self.num_for_avg / self.batch_count, "Epoch avg " + self.name)
        print("Average {} in epoch = {}".format(self.name,self.num_for_avg / self.batch_count))
        self.num_for_avg = 0

        plt.plot(self.array_epoch, self.array_loss)
        plt.title(self.name)
        plt.xlabel("Epoch")
        plt.ylabel(self.name)
        plt.savefig(self.name+".png")
        plt.show()