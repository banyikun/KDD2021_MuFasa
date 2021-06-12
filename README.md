### Multi-facet Contextual Bandits

Contextual multi-armed bandit has shown to be an effective tool in recommender systems. 
In this paper, we study a novel problem of multi-facet bandits involving a group of bandits, each characterizing the users' needs from one unique aspect. 
In each round, for the given user, we need to select one arm from each bandit, such that the combination of all arms maximizes the final reward. 
This problem can find immediate applications in E-commerce, healthcare, etc. 
To address this problem, we propose a novel algorithm, named MuFasa, which utilizes an assembled neural network to jointly learn the underlying reward functions of multiple bandits. 
It estimates an Upper Confidence Bound (UCB) linked with the expected reward to balance between exploitation and exploration. 
Under mild assumptions, we provide the regret analysis of MuFasa. It can achieve the near-optimal $\widetilde{ \mathcal{O}}((K+1)\sqrt{T})$ regret bound where $K$ is the number of bandits and $T$ is the number of played rounds. 
Furthermore, we conduct extensive experiments to show that MuFasa outperforms strong baselines on real-world data sets.



## Upper Confidence Bound

![alt text](./MuFasa_Readme.jpg) 

