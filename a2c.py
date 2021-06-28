from actorcritic import ActorCritic


class A2C(ActorCritic):
    def __init__(self, args):
        args.ppo_episodes = 1
        args.batch_size   = 1
        ActorCritic.__init__(self, args)

    def get_actor_loss(self, current_log_probs, log_probs, A_k, entropy):
        return (-current_log_probs * A_k).mean() - self.args.noise * entropy

    def get_critic_loss(self, V, rewards, discounted_return):
        return self.mse(V, discounted_return)
