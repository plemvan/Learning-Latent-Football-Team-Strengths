# Learning-Latent-Football-Team-Strengths
Predicting sports outcomes is yet a classic research problem in statistics and machine learning. Traditional supervised learning approaches require an explicit target variable(e.g. number of goals, rating, etc...). However, in many real-world contexts, true performance levels are latent and can only be inferred indirectly through pairwise outcomes (”Team A beats Team B”).

The Bradley–Terry (BT) model provides a probabilistic framework to infer these latent strengths by modelling the probability of one competitor beating another as a logistic function of their respective abilities. Yet, the standard BT model assumes fixed parameters and cannot incorporate contextual information ore generalize to unseen competitors.

This project aims to first reproduce and analyse the classical BT model, and then extend it to a neural formulation inspired by the Neural Bradley–Terry Rating (NBTR) framework. It performs implicit supervision, learning latent strengths solely from observed match outcomes without access to explicit target values. The goal is to implement and empirically evaluate these models using real sports data.
