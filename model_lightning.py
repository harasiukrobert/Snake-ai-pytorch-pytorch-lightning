import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from helper import plot
from game import SnakeGameAI, Direction, Point
from agent2 import Agent
from dataset_iterable import SnakeGameDataset

torch.set_float32_matmul_precision('high')

class Linear_QNetwork_lightning(pl.LightningModule):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.agent = Agent()
        self.game = SnakeGameAI()
        self.dataset = SnakeGameDataset(self.agent, self.game, self)

        self.record = 0
        self.plot_scores = []
        self.plot_mean_scores = []
        self.total_score = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=None, num_workers=0)

    def end_game_check(self,done,score):
        self.agent.train_long_memory(self)
        self.game.reset()
        self.agent.n_games += 1

        if score > self.record:
            self.record = score

        print(f'Game: {self.agent.n_games} | Score: {score} | Record: {self.record}')

        self.plot_scores.append(score)
        self.total_score += score
        mean_score = self.total_score / self.agent.n_games
        self.plot_mean_scores.append(mean_score)
        plot(self.plot_scores, self.plot_mean_scores)

    def training_step(self, batch, batch_idx):
         self.optimizer = self.optimizers()
         state_old, final_move, reward, state_new, done = batch

         state = torch.tensor(state_old, dtype=torch.float)
         next_state = torch.tensor(state_new, dtype=torch.float)
         action = torch.tensor(final_move, dtype=torch.long)
         reward = torch.tensor(reward, dtype=torch.float)

         if len(state.shape) == 1:
             state = torch.unsqueeze(state, 0)
             next_state = torch.unsqueeze(next_state, 0)
             action = torch.unsqueeze(action, 0)
             reward = torch.unsqueeze(reward, 0)
             done = (done,)

         pred = self(state)
         target = pred.clone()
         for idx in range(len(done)):
             Q_new = reward[idx]
             if not done[idx]:
                 Q_new = reward[idx] + 0.9 * torch.max(self(next_state[idx]))

             target[idx][torch.argmax(action[idx]).item()] = Q_new

         loss = self.criterion(target, pred)
         return loss

if __name__ == '__main__':
    trainer = pl.Trainer(max_epochs=1, fast_dev_run=False)
    model = Linear_QNetwork_lightning(11,256,3)
    trainer.fit(model)
