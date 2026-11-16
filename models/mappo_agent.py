import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

class NeighborAttentionEncoder(nn.Module):
    """
    Permutation-invariant self-attention encoder for neighbor observations.
    Encodes K-nearest neighbor tokens so policy does not depend on arbitrary ordering.
    """
    def __init__(self, neighbor_feat_dim=7, embed_dim=32):
        super().__init__()
        self.proj = nn.Linear(neighbor_feat_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, neighbor_tokens):
        # neighbor_tokens: [batch_size, K, neighbor_feat_dim]
        x = self.proj(neighbor_tokens)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        # Max-pool across neighbors for permutation invariance
        pooled = torch.max(x, dim=1)[0]
        return pooled

class Decentralized3DActor(nn.Module):
    """
    Decentralized 3D Actor Network:
    Takes 3D local observations (self 3D kinematics, neighbor attention embedding, 3D survivor radar)
    and predicts continuous 3D thrust commands [ax, ay, az].
    """
    def __init__(self, k_neighbors=4, neighbor_feat_dim=7, action_dim=3, hidden_dim=128):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.neighbor_feat_dim = neighbor_feat_dim
        self.self_dim = 6     # [x, y, z, vx, vy, vz]
        self.survivor_dim = 4 # [rel_x, rel_y, rel_z, detected]
        self.embed_dim = 32

        # Attention encoder for neighbors
        self.neighbor_encoder = NeighborAttentionEncoder(neighbor_feat_dim, embed_dim=self.embed_dim)

        # Trunk MLP
        combined_dim = self.self_dim + self.embed_dim + self.survivor_dim
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def _split_obs(self, obs):
        batch_size = obs.shape[0]
        self_feat = obs[:, :self.self_dim]
        neighbor_flat = obs[:, self.self_dim : self.self_dim + (self.k_neighbors * self.neighbor_feat_dim)]
        neighbor_tokens = neighbor_flat.view(batch_size, self.k_neighbors, self.neighbor_feat_dim)
        survivor_feat = obs[:, -self.survivor_dim:]
        return self_feat, neighbor_tokens, survivor_feat

    def forward(self, obs):
        self_feat, neighbor_tokens, survivor_feat = self._split_obs(obs)
        neighbor_emb = self.neighbor_encoder(neighbor_tokens)
        fused = torch.cat([self_feat, neighbor_emb, survivor_feat], dim=-1)

        feat = self.net(fused)
        mu = torch.tanh(self.mu_head(feat))
        std = torch.exp(torch.clamp(self.log_std, -2.0, 0.5))
        return mu, std

    def get_action(self, obs, deterministic=False):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_action(self, obs, action):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

class Centralized3DCritic(nn.Module):
    """
    Centralized 3D Critic (CTDE):
    Evaluates full global 3D swarm state during training.
    """
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, global_state):
        return self.net(global_state)

class MAPPOAgent:
    """
    MAPPO container for 3D Drone Swarm navigation.
    """
    def __init__(
        self,
        k_neighbors=4,
        state_dim=152,
        action_dim=3,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.actor = Decentralized3DActor(k_neighbors=k_neighbors, action_dim=action_dim).to(self.device)
        self.critic = Centralized3DCritic(state_dim=state_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_actions(self, observations_dict, deterministic=False):
        self.actor.eval()
        actions = {}
        log_probs = {}
        with torch.no_grad():
            for agent, obs in observations_dict.items():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                act_t, logp_t = self.actor.get_action(obs_t, deterministic=deterministic)
                actions[agent] = act_t.squeeze(0).cpu().numpy()
                log_probs[agent] = logp_t.squeeze(0).item()
        return actions, log_probs

    def get_value(self, global_state):
        self.critic.eval()
        with torch.no_grad():
            state_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.critic(state_t).squeeze(0).item()

    def update(self, rollouts, ppo_epochs=5, batch_size=128):
        self.actor.train()
        self.critic.train()

        obs_t = torch.as_tensor(rollouts["obs"], dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(rollouts["actions"], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(rollouts["log_probs"], dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(rollouts["advantages"], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(rollouts["returns"], dtype=torch.float32, device=self.device)
        g_state = torch.as_tensor(rollouts["global_states"], dtype=torch.float32, device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n_samples = len(obs_t)
        actor_losses, critic_losses = [], []

        for _ in range(ppo_epochs):
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                b_idx = indices[start : start + batch_size]

                # Actor Loss
                logp, entropy = self.actor.evaluate_action(obs_t[b_idx], act_t[b_idx])
                ratio = torch.exp(logp - old_logp[b_idx])
                surr1 = ratio * adv[b_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv[b_idx]
                p_loss = -torch.min(surr1, surr2).mean() - (self.entropy_coef * entropy.mean())

                self.actor_optimizer.zero_grad()
                p_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                actor_losses.append(p_loss.item())

                # Critic Loss
                vals = self.critic(g_state[b_idx]).squeeze(-1)
                v_loss = nn.functional.mse_loss(vals, ret[b_idx])

                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                critic_losses.append(v_loss.item())

        return float(np.mean(actor_losses)), float(np.mean(critic_losses))

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path, load_critic=True):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        if load_critic and "critic" in ckpt:
            try:
                self.critic.load_state_dict(ckpt["critic"])
            except RuntimeError as e:
                print(f"[NOTE] Decentralized Actor loaded successfully. Critic skipped (expected for variable swarm size): {e}")
