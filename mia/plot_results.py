import matplotlib.pyplot as plt
plt.plot([500, 1000, 2500, 5000, 10000, 20000, 30000, 40000],[0.140, 0.328, 0.465, 0.576, 0.649, 0.663, 0.680, 0.731], marker = 'o')
plt.plot([500, 1000, 2500, 5000, 10000, 20000, 30000, 40000],[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
#plt.plot([1000, 3000, 5000, 10000], [0.649, 0.358, 0.261, 0.199], marker= 'o')
#plt.plot([1000, 3000, 5000, 10000],[0.1, 0.1, 0.1, 0.1])
plt.ylim(0,1)
plt.ylabel('accuracy')
#plt.xlabel('training set size')
#plt.title('Membership Inference Attack on TSG with different training set sizes')
plt.xlabel('epochs')
plt.title('Membership Inference Attack on TSG trained with 1.000 trajectories')
plt.savefig("mia_1000.pdf")