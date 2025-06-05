import game.wrapped_flappy_bird as fb_game
import pygame
from pygame.locals import *
import sys
from tqdm.auto import trange

import torch
import torch.nn as nn

import numpy as np
import wandb
import os



def main():
    action_terminal = fb_game.GameState()

    while True:
        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        input_actions = [1, 0]
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                input_actions = [0, 1]
            else:
                input_actions = [1, 0]

        action_terminal.frame_step(input_actions)

class FlappyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 80),
            nn.ReLU(),
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def auto(policy, epochN, trainN=1000):
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    game = fb_game.GameState()

    tr = trange(trainN)
    for episode in tr:
        log_probs = []
        entropies = []
        actions = []
        rewards = []

        done = False
        state = game.player_state()
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            logits = torch.clamp(policy(state_tensor), -10, 10)
            m = torch.distributions.Bernoulli(logits=logits)
            action = m.sample()
            log_prob = m.log_prob(action)
            log_probs.append(log_prob)
            entropy = m.entropy()
            entropies.append(entropy)

            act = int(action.item())
            actions.append(act)
            frame = [act, 1 - act]
            _, reward, done = game.frame_step([act, 1 - act])
            flap_penalty = -0.05 if act == 0 else 0.0
            reward += flap_penalty
            rewards.append(reward)
            state = game.player_state()

        returns = compute_returns(rewards)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        entropy = torch.stack(entropies).sum()
        loss = sum(-lp * R for lp, R in zip(log_probs, returns))
        loss += -0.1 * entropy
        mean_action = torch.tensor(actions, dtype=torch.float32).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr.set_description(f"Epoch {epochN} | Episode {episode} | Reward: {sum(rewards)} | Loss: {loss}")
        log_dict = {
            "Train Loss": loss,
            "Reward": sum(rewards),
            "Episode": epochN*trainN + episode,
        }
        wandb.log(log_dict)


    save_path = f"./checkpoints_with_vy/policy_{epochN}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(policy.state_dict(), save_path)
    wandb.save(save_path)
    print(f"Model saved to {save_path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    print("Model loaded successfully.")
    return model

def train(epochN=27):
    wandb.init(
        project="hw6",
        name="new rewards",
        config={
            "epochs": epochN
        }
    )

    policy = FlappyPolicy()
    for epoch in range(epochN):
        auto(policy, epoch)





if __name__ == '__main__':
    train()
