import torch

from tqc.functions import quantile_huber_loss_f
from tqc import DEVICE

###debug

class Trainer(object):
	def __init__(
		self,
		*,
		actor,
		critic,
		critic_target,
		discount,
		tau,
		top_quantiles_to_drop,
		target_entropy,
		n_nets,
		beta,
		n_quantiles,
	):
		self.actor = actor
		self.critic = critic
		self.critic_target = critic_target
		self.log_alpha = torch.zeros((1,), requires_grad=True, device=DEVICE)
		self.n_nets = n_nets

		# TODO: check hyperparams
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

		self.discount = discount
		self.tau = tau
		self.top_quantiles_to_drop = top_quantiles_to_drop
		self.target_entropy = target_entropy
		self.beta = beta
		self.n_quantiles = n_quantiles

		self.quantiles_total = critic.n_quantiles * critic.n_nets

		self.total_it = 0


	def select_critic(self, q, num):
		q_var = q.var(2)
		q_mean = q.mean(2)
		a = q_var.argmin(1).reshape(256,-1)
		#a = a.reshape(256, -1)
		b = q_mean.argmin(1).reshape(256,-1)
		#b = b.reshape(256, -1)
		d = torch.cat((a, b), 1)
		g = d.reshape(512,1)
		c = q_mean.var(1)
		c = (c > self.beta).long()
		d = d.reshape(512, 1)[torch.arange(256).cuda() * 2 + c]
		d = d.squeeze()
		target = q.reshape(-1, num)[torch.arange(256).cuda() * self.n_nets + d]
		return target

	def train(self, replay_buffer, batch_size=256):
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		alpha = torch.exp(self.log_alpha)

		# --- Q loss ---
		with torch.no_grad():
			# get policy action
			new_next_action, next_log_pi = self.actor(next_state)

			# compute and cut quantiles at the next state
			next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
			next_z = self.select_critic(next_z, self.n_quantiles)

			# compute target
			target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)

		cur_z = self.critic(state, action)

		critic_loss = quantile_huber_loss_f(cur_z, target)

		# --- Policy and alpha loss ---
		new_action, log_pi = self.actor(state)
		alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
		actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

		# --- Update ---
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		self.total_it += 1

	def save(self, filename):
		filename = str(filename)
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_target.state_dict(), filename + "_critic_target")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
		torch.save(self.log_alpha, filename + '_log_alpha')
		torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

	def load(self, filename):
		filename = str(filename)
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_target.load_state_dict(torch.load(filename + "_critic_target"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.log_alpha = torch.load(filename + '_log_alpha')
		self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))
