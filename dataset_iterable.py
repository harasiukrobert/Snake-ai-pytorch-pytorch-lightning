from torch.utils.data import DataLoader, IterableDataset


class SnakeGameDataset(IterableDataset):
    def __init__(self, agent, game, model):
        super().__init__()
        self.agent = agent
        self.game = game
        self.model = model

    def __iter__(self):
        while True:
            state_old = self.agent.get_state(self.game)
            final_move = self.agent.get_action(state_old, self.model)
            reward, done, score = self.game.play_step(final_move)
            state_new = self.agent.get_state(self.game)
            self.agent.remember(state_old, final_move, reward, state_new, done)

            yield state_old, final_move, reward, state_new, done
            if done:
                self.model.end_game_check(done, score)