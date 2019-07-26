import numpy as np
import tensorflow as tf

class PolicyGradient:
  def __init__(
          self,
          state_dims,
          action_dims,
          learning_rate=0.01,
          reward_decay=0.95,
          output_graph=False
  ):
    self.state_dims = state_dims
    self.action_dims = action_dims
    self.lr = learning_rate
    self.gamma = reward_decay
    
    self.memory = []

    self._build_net()

    self.sess = tf.Session()

    if output_graph:
      tf.summary.FileWriter('logs/', self.sess.graph)

    self.sess.run(tf.global_variables_initializer())

  def _build_net(self):
    self.s = tf.placeholder(tf.float32, [None, self.state_dims], name='state')
    self.a = tf.placeholder(tf.int32, [None, ], name='actions')
    self.vt = tf.placeholder(tf.float32, [None, ], name='actions_value')
    with tf.variable_scope('eval_net'):
      h1 = tf.layers.dense(self.s, 10, activation=tf.nn.relu)
      all_act = tf.layers.dense(h1, self.action_dims, activation=None)

    self.all_act_prob = tf.nn.softmax(all_act)

    with tf.variable_scope('loss'):
      neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.a)
      # neg_log_prob = reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.a, self.action_dims), axis=1)
      self.loss = tf.reduce_mean(neg_log_prob * self.vt)
    with tf.variable_scope('train'):
      self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

  def store_transition(self, s, a, r):
    self.memory.append( (s, a, r) )

  def choose_action(self, state, _eval=False):
    state = state[np.newaxis, :]

    action_prob = self.sess.run(self.all_act_prob, feed_dict={
                      self.s: state })
    action = np.random.choice(self.action_dims, p=action_prob[0])

    return action

  def learn(self):
    discounted_ep_rs = self._discount_and_norm_reward()
    s = [ m[0] for m in self.memory ]
    a = [ m[1] for m in self.memory]
    
    self.sess.run(self.train_op, feed_dict={
        self.s: s,
        self.a: a,
        self.vt: discounted_ep_rs })

    self.memory = []

  def _discount_and_norm_reward(self):
    discounted_ep_rs = np.zeros(len(self.memory))
    cummluative_reward = 0
    for t in reversed(range(0, len(self.memory))):
      cummluative_reward = self.memory[t][2] + cummluative_reward * self.gamma
      discounted_ep_rs[t] = cummluative_reward

    discounted_ep_rs -= np.mean(discounted_ep_rs)
    discounted_ep_rs /= np.std(discounted_ep_rs)

    return discounted_ep_rs
