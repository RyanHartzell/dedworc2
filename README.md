# dedworc2
dedworc2 - DEnsity-based Drone Workflow for Observation and Reinforcement-Learning of Crowd Characteristics (Too Crowded, backwards)

## Authors
Ryan Hartzell |
Matt Desaulniers

## Overview
This project will eventually allow for the following use and functionality:

* Run crowd simulations (or basic particle simulations)
* Simulate observation of crowd via a distributed network of drones
* Create kernel density estimations over 2D spaces, and construct crowd/particle dynamics related metrics and heuristics from observations
* Train a Pytorch reinforcement learning model to observe a dynamic crowd in a fixed space
* Test and Evaluate learned model
* (ULTIMATE GOAL) Run the learned model and display a simulation (and performance metrics) where all individual agents work collaboratively to navigate the environment and sense the underlying crowd's dynamics efficiently, tracking regions of interest or events of interest in the crowd over time


## Requirements

### Crowd Simulator

* Must be Closed Environment
* Must have particles objects
* Particle objects must have dynamics (e.g. attraction, repulsion forces, etc.)
* Must have a visualizer
* Must have some form of observability
* Must be able to check boundaries against a point
* Must be integrated with GYM interface in pytorch

### Drone (Agents)

* Must be able to move within boundary of environment
* Must be able to observe local crowd state
* Must be able to build internal map of global environment
* Must own actor/critic modules
* Must be able to visualize map
    * Traces
    * Current Crowd Belief state
* Must have Random Walk mode
* Must be able to flush  history to disk

