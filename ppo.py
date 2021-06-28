import torch

from actorcritic import ActorCritic


class PPO(ActorCritic):
    def __init__(self, args):
        ActorCritic.__init__(self, args)

    def get_actor_loss(self, current_log_probs, log_probs, A_k, entropy):
        # Calculate ratios
        ratios = torch.exp(current_log_probs - log_probs)

        # Calculate surrogate losses
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1-self.args.clip, 1+self.args.clip) * A_k

        # Calculate actor and critic loss
        actor_loss = -torch.min(surr1, surr2).mean() - self.args.noise * entropy

        return actor_loss

    def get_critic_loss(self, V, rewards, discounted_return):
        return self.mse(V, discounted_return)