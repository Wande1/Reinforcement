# MDP APPROACH FOR 2 X 2 GRID EXAMPLE
Example; Learning an agent travelling through a 2*2 grid (4 states)

Red wall prevents direct moves from S1 to S2

States S1, S2, S3 give reward of -1; state S4 gives reward of +10

Action: Left, Right Up, and Down


#LIBRARY
library(MDPtoolbox)
library(devtools)
library(ReinforcementLearning)


# 1. Defining the Set of Actions - Left, Right, Up and Down for 2 x 2 matrix

#Up Action
up=matrix(c( 1, 0, 0, 0,
             0.7, 0.2, 0.1, 0,
             0, 0.1, 0.2, 0.7,
             0, 0, 0, 1),
          nrow=4,ncol=4,byrow=TRUE)




#Down Action
down=matrix(c(0.3, 0.7, 0, 0,
              0, 0.9, 0.1, 0,
              0, 0.1, 0.9, 0,
              0, 0, 0.7, 0.3),
            nrow=4,ncol=4,byrow=TRUE)





#Left Action
left=matrix(c( 0.9, 0.1, 0, 0,
               0.1, 0.9, 0, 0,
               0, 0.7, 0.2, 0.1,
               0, 0, 0.1, 0.9),
            nrow=4,ncol=4,byrow=TRUE)




#Right Action
right=matrix(c( 0.9, 0.1, 0, 0,  0.1, 0.2, 0.7, 0,
                0, 0, 0.9, 0.1,
                0, 0, 0.1, 0.9),
             nrow=4,ncol=4,byrow=TRUE)






#Aggregate previous matrices to create transistion probabilities into list T
T <- list(up=up, left=left, down=down, right=right)
T
#T = Transistion probability




#Create matrix with rewards:

R=matrix(c( -1, -1, -1, -1,
            -1, -1, -1, -1,
            -1, -1, -1, -1,
            10, 10, 10, 10),
         nrow=4,ncol=4,byrow=TRUE)

#R = Reward matrix



#Check if this provides a well-defined MDP
mdp_check(T, R) # empty string ==> ok


#Policy iteration with discount factor g = 0.9
m <- mdp_policy_iteration(P=T,
                          R=R,
                          discount=0.9)


#Display optimal policy p
m$policy

names(T)[m$policy]

#Display value funtion vp
m$V
#The value shows the movement of following this policy as I move from state to state.


#REINFORCEMENT LEARNING

# Viewing the pre-built function for each state, action and reward

env <- gridworldEnvironment
print(env)

states <- c("S1", "S2", "S3", "S4")
states
actions <- c("up", "down", "left", "right")
actions


# Sample N = 1000 random sequences from the environment

#Data format must be (s, a, r,s_new) tuples
#as row in a dataframe structure.

data <- sampleExperience(N = 1000,
                         env = env,
                         states = states,
                         actions = actions)


head(data)


# Define reinforcement learning parameter
control <- list(alpha = 0.1, #low learning rate
                gamma = 0.5, #middle discount factor
                epsilon = 0.1) #low exploration factor

control



#MODEL

model <- ReinforcementLearning(data, 
                               s = "State",
                               a = "Action",
                               r = "Reward",
                               s_new = "NextState",
                               control = control)

#Print result
print(model)


##CONCLUSION***






