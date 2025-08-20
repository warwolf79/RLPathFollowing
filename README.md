# RLPathFollowing
Double Integrator RL agent Path Following 

## Project Description 
I have implemented a double integrator dynamics (as a quadrotor) for path following with a PD controller and obstacle avoidance with the potential field method. The agent is a point mass system with a double integrator dynamics, and it should follow the reference path while avoiding static and dynamic obstacles. The simulation is done in Simulink. But the reference path is a .csv file generated from a Python script. Since my simulation step size is fixed at 0.01 seconds, the reference path step size (dt) is also set to 0.01 seconds. 

It includes three phases: 

1- vertical climb from zero origin to 5 meters. (and a transition phase to move smoothly to the next phase) 

2- cruise path along the x-axis for a while at the same altitude. 

3- Helical maneuver along the z-axis. 


There are three .csv files generated as the reference path, incrementing the complexity level of the reference path.
The first complexity level includes the first phase of the reference path ( this first phase includes 701 points [700 steps] in total), complexity 2 includes phases one and two (8812 points [8811 steps]), and the third complexity level includes all three phases (13501 points) (The whole simulation time is 135 seconds).
The third .csv file is imported as the reference path in Simulink with an Import block.

I want to do the same path-following with a Reinforcement learning algorithm, probably TD3, for my complex environment; this is why I divided my reference path into three complexity levels, to implement a curriculum training and transfer the learned data to the next complexity level using the same agent. 

The RL agent should output the 3d acceleration vector to follow the path, and after training, I need a .csv file of the followed path data to import into my Simulink model. 

I should mention that the maximum acceleration of the reference path is 2.8 m/s^2. So the action should be bound to a proper value.
