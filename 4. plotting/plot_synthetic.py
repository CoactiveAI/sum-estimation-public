import numpy as np
import matplotlib.pyplot as plt

class SyntheticTask:
    def __init__(self,n,seed,max_k = 1000):
        self.n = n
        self.max_k = max_k

        self.items_per_level = {}

        rand_gen = np.random.default_rng(seed=seed)

        min_remaining_level = 1
        cut_levels = []
        p = 1

        loc = 0
        while loc < n:
            while True:
                level = rand_gen.geometric(0.5) + (min_remaining_level-1)
                if level not in cut_levels:
                    break
            
            if level not in self.items_per_level:
                self.items_per_level[level] = []
            self.items_per_level[level].append(loc)

            if len(self.items_per_level[level]) >= max_k:
                cut_levels.append(level)
                p -= 2**(-level)
                while min_remaining_level in cut_levels:
                    min_remaining_level += 1

            loc += rand_gen.geometric(p)


    def GetU(self,k):
        assert(k <= self.max_k)
        items = []
        for level in self.items_per_level.keys():
            for item in self.items_per_level[level][:k]:
                items.append( (item, level) )
        return items
    
    def GetNonFullLevels(self,k):
        assert(k <= self.max_k)
        items = []
        for level in self.items_per_level.keys():
            if len(self.items_per_level[level]) < k:
                items += self.items_per_level[level]
        return items
    
def Estimate(U, k, n, num_nonzero, non_full):
    median_non_full = np.median(non_full)
    back_half = [num for num in non_full if num >= median_non_full]
    c = np.mean([int(num < num_nonzero) for num in back_half])

    E = 0
    p = 1
    level_to_count = {}

    for index, level in sorted(U):
        f_value = int(index < num_nonzero) - c
        E += f_value / p
        
        if level not in level_to_count:
            level_to_count[level] = 0
        level_to_count[level] += 1

        if level_to_count[level] == k:
            p -= 2**(-level)

    return E + c*n

plot_format = "pdf"
n = 10**7
k=200
x_values = [int(10**power) for power in np.arange(0.0,7.1,0.1)]

rel_errs_list = []
for seed in range(1000):
    print("seed",seed)
    task = SyntheticTask(n,seed)
    U = task.GetU(k)
    non_full = task.GetNonFullLevels(k)

    rel_errs = []
    for num_nonzero in x_values:
        rel_errs.append( abs(Estimate(U,k,n,num_nonzero,non_full) - num_nonzero)/num_nonzero )
    rel_errs_list.append(rel_errs)

rel_errs_matrix = np.array(rel_errs_list)

sorted_rel_errs_matrix = np.sort(rel_errs_matrix,axis=0)
centers = sorted_rel_errs_matrix[949,:]
lowers = sorted_rel_errs_matrix[936,:] #Pr(Binomial(1000,0.95) <= 936) = 2.843%
uppers = sorted_rel_errs_matrix[963,:] #Pr(Binomial(1000,0.95) >= 964) = 2.115%


plt.plot(x_values,centers,label="Our algorithm")
plt.fill_between(x_values, lowers, uppers,alpha=0.3)
plt.plot(x_values,len(x_values)*[0.1631],label="Our analysis",c="r")

plt.xlabel("Number of non-zero f values")
plt.ylabel('95th-percentile Relative Error')
plt.xscale('log')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(f"plots/synthetic.{plot_format}", format=plot_format, bbox_inches="tight")

