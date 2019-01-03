################## Global Settings: ##################
#Fixed settings
LAYERS = 13  #Upgrade layers merged: layers -2 (Hard coded).

### Changeable settings

EPISODES = 1000000
LR = 1e-3
RANDOMSTART = True  # randomize agent starting location.
CENTERED    = True  # reduces layers by 1 (my position)
BLASTPAT    = True  # reduces layers by 2 (bomb map, bomb strength, bomb life is merged)

# reward shaping:
SAREWARD    = True   # reward shaping using Simple Agent (Off policy, Simple agent in control)
PXREWARD    = False  # Reward shaping using the Pommerman X (link) reward shaping

#select network:
CONVNET          = False # convolutional
TENSORFLOWRESNET = False  # layers -3 (ammo, blast_strength & can_kick moved to singular values)
TFRWIGHTMAN      = True


#Select algorithm:
PPO = True
DQN = False

#Exploration (off policy)
EXPLORE = False

#BatchNorm Hyperparams:
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

#Calculate model layers based on params
if CENTERED:
    LAYERS -= 1
if BLASTPAT:
    LAYERS -= 2

################## IMPORTS ##################

import os
import sys
import numpy as np

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.characters import Bomber
from pommerman import utility
from pommerman.constants import *
from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.agents import PPOAgent
from tensorforce.core.networks.network import LayerBasedNetwork
from tensorforce.core.networks.layer import TFLayer
from tensorforce.core.networks import Layer
from tensorforce.core.networks import Network
from tensorflow import nn
import tensorflow as tf
import random
import csv
import copy
       

os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
################## TENSORFLOW RESNET NETWORK ##################
#inspired by:
#https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

#res-net function with batch nomalization
def res(inputs, filters, training, strides):
	shortcut = inputs
	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=3, strides=strides)
	inputs = tf.layers.batch_normalization(inputs=inputs, axis=3,momentum=_BATCH_NORM_DECAY,epsilon=_BATCH_NORM_EPSILON, center=True,scale=True, training=training, fused=True)
	inputs = tf.nn.relu(inputs)
	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=3, strides=1)
	inputs = tf.layers.batch_normalization(inputs=inputs, axis=3,momentum=_BATCH_NORM_DECAY,epsilon=_BATCH_NORM_EPSILON, center=True,scale=True, training=training, fused=True)	
	inputs += shortcut
	inputs = tf.nn.relu(inputs)
	return inputs

#res-net function with batch nomalization
#no shortcut, for input data
def resnsc(inputs, filters, training, strides):
	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=3, strides=strides)
	inputs = tf.layers.batch_normalization(inputs=inputs, axis=3,momentum=_BATCH_NORM_DECAY,epsilon=_BATCH_NORM_EPSILON, center=True,scale=True, training=training, fused=True)
	inputs = tf.nn.relu(inputs)
	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=3, strides=1)
	inputs = tf.layers.batch_normalization(inputs=inputs, axis=3,momentum=_BATCH_NORM_DECAY,epsilon=_BATCH_NORM_EPSILON, center=True,scale=True, training=training, fused=True)	
	inputs = tf.nn.relu(inputs)
	return inputs

#res-net function without batch nomalization
def resnbn(inputs, filters, training, strides):
	shortcut = inputs
	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=3, strides=strides)
	inputs = tf.nn.relu(inputs)
	inputs = conv2d_fixed_padding(
			inputs=inputs, filters=filters, kernel_size=3, strides=1)
	inputs += shortcut
	inputs = tf.nn.relu(inputs)
	return inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
	"""Strided 2-D convolution with explicit padding."""
	# The padding is consistent and is based only on `kernel_size`, not on the
	# dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
	if strides > 1:
		inputs = fixed_padding(inputs, kernel_size)

	return tf.layers.conv2d(
			inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
			padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
			kernel_initializer=tf.variance_scaling_initializer())

class TFNetworkRES(Network):
		def __init__(self, scope='TFNetworkRES', summary_labels=()):
				super(TFNetworkRES, self).__init__(
						scope=scope, summary_labels=summary_labels)	
		def tf_apply(self, x, internals, update, return_internals=False):
				x1 = x['state']
                #get singular values from whole boards:
				kick = tf.contrib.layers.flatten(tf.slice(x1, [0, 0, 0, (LAYERS-1)], [-1, 1, 1, 1]))
				blast = tf.contrib.layers.flatten(tf.slice(x1, [0, 0, 0, (LAYERS-2)], [-1, 1, 1, 1]))
				ammo = tf.contrib.layers.flatten(tf.slice(x1, [0, 0, 0, (LAYERS-3)], [-1, 1, 1, 1]))
                #cut off whole boards
				state = tf.slice(x1, [0, 0, 0, 0], [-1, 11, 11, (LAYERS-3)])
                #res-net
				state = conv2d_fixed_padding(state, 64, 3, 1)
				state = resnsc(state, 64, True, 1)
				state = res(state, 64, True, 1)
				state = res(state, 64, True, 1)
				state = res(state, 64, True, 1)
				state = res(state, 64, True, 1)
				state = res(state, 64, True, 1)
				state = res(state, 64, True, 1)
				state = res(state, 64, True, 1)
				state = res(state, 64, True, 1)
				state = res(state, 64, True, 1)
				state = tf.contrib.layers.flatten(state)
                ## add singular values before dense layer
				state = tf.concat([state, kick,blast,ammo], axis=1)
				state = tf.layers.dense(state,3000,activation=tf.nn.relu)
				state = tf.layers.dense(state,512,activation=tf.nn.relu)
				state = tf.nn.softmax(state)

				# Combination
				if return_internals:
						return	state, list() #tf.multiply(image, caption), list()
				else:
						return	state # tf.multiply(image, caption)
					

#Network architecture based on rwightman https://github.com/rwightman/pytorch-pommerman-rl                    

class TFNetworkRWI(Network):

		def __init__(self, scope='TFNetworkRWI', summary_labels=()):
				super(TFNetworkRWI, self).__init__(
						scope=scope, summary_labels=summary_labels)	
		def tf_apply(self, x, internals, update, return_internals=False):
				x1 = x['state']
                #get singular values from whole boards:
				kick = tf.contrib.layers.flatten(tf.slice(x1, [0, 0, 0, (LAYERS-1)], [-1, 1, 1, 1]))
				blast = tf.contrib.layers.flatten(tf.slice(x1, [0, 0, 0, (LAYERS-2)], [-1, 1, 1, 1]))
				ammo = tf.contrib.layers.flatten(tf.slice(x1, [0, 0, 0, (LAYERS-3)], [-1, 1, 1, 1]))
                #cut off whole boards
				state = tf.slice(x1, [0, 0, 0, 0], [-1, 11, 11, (LAYERS-3)])
                
				state = tf.layers.conv2d(state, 64, 3, 1,'SAME')  #11x11
				state = tf.layers.conv2d(state, 64, 3, 1,'SAME')  #11x11
				state = tf.layers.conv2d(state, 64, 3, 1,'VALID') #9x9
				state = tf.layers.conv2d(state, 64, 3, 1,'VALID') #7x7
				state = tf.contrib.layers.flatten(state)
				state = tf.layers.dense(state,1024,activation=tf.nn.relu)     
				state = tf.layers.dense(state,512,activation=None)                    

				numeric = tf.concat([kick,blast,ammo], axis=1)   
				numeric = tf.layers.dense(numeric,128,activation=tf.nn.relu)                 
				numeric = tf.layers.dense(numeric,128,activation=tf.nn.relu) 
                
				state = tf.concat([state,numeric], axis=1)
				# Combination
				if return_internals:
						return	state, list() #tf.multiply(image, caption), list()
				else:
						return	state # tf.multiply(image, caption)
					                    
                    
                    
################## Init Environment ##################

config = ffa_v0_fast_env()
env = Pomme(**config["env_kwargs"])
env.seed(0)
                    
    
################## Network type: ##################    

if CONVNET:
	net =  [dict(type='conv2d', size=64, window=3, stride=1, padding='SAME', activation='relu'),
			dict(type='conv2d', size=64, window=3, stride=1, padding='SAME', activation='relu'),
			dict(type='conv2d', size=64, window=3, stride=1, padding='SAME', activation='relu'),
			dict(type='conv2d', size=64, window=3, stride=1, padding='SAME', activation='relu'),
			dict(type='flatten'),
			dict(type='dense', size = 7744),
			dict(type='dense', size = 512),        
			dict(type='nonlinearity', name='softmax')]     
            
if TENSORFLOWRESNET:
    net = TFNetworkRES

if TFRWIGHTMAN:
    net = TFNetworkRWI    
    
    
######## Exploration ########    
    
if EXPLORE:
	 ex = dict(
			type= 'epsilon_anneal', #'epsilon_decay',
			initial_epsilon=1.0,
			final_epsilon=0.001,
			timesteps= (int(EPISODES * 0.7))
			)
else:
	 ex = None

    
################## Learning Algorithms: ##################               
  
if PPO:
	agent = PPOAgent(
	states=dict(type='float', shape=(11,11,LAYERS)),
	actions=dict(type='int',	num_actions=env.action_space.n),
	batching_capacity=1000,
	network=net, #CustomTFLayerNetwork,
	actions_exploration=ex,
	step_optimizer=dict(
		type='adam',
		learning_rate=LR
		)
    )

        

if DQN:
	agent = DQNAgent(
	states=dict(type='float', shape=(11,11,LAYERS)),
	actions=dict(type='int', num_actions=env.action_space.n),
	network=net,
	batching_capacity=1000,
	actions_exploration=ex,
	optimizer=dict(
		type='adam',
		learning_rate=LR
		)        
	)
       

################## Initiate Tensorforce Agent ##################
class TensorforceAgent(BaseAgent):
	def act(self, obs, action_space):
		pass

###################### Set Agent ###################

# Add 3 agents
agents = []

agents.append(SimpleAgent(config["agent"](0, config["game_type"])))
agents.append(RandomAgent(config["agent"](1, config["game_type"])))
agents.append(RandomAgent(config["agent"](2, config["game_type"])))

# Add Traning Agent
agent_id = 3
agents.append(TensorforceAgent(config["agent"](agent_id, config["game_type"])))

# Add all agents to the environment
env.set_agents(agents)

# Define the agent that is trained
env.set_training_agent(agents[agent_id].agent_id)

# initiate game
env.set_init_game_state(None)



################## REWARD SHAPING - Pommerman X ##################
#source Pommerman-x - https://github.com/papkov/pommerman-x
def isBetween(a, b, c):
		crossproduct = (c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y)
		epsilon = 0.0001
		# compare versus epsilon for floating point values, or != 0 if using integers
		if abs(crossproduct) > epsilon:
				return False

		dotproduct = (c.x - a.x) * (b.x - a.x) + (c.y - a.y)*(b.y - a.y)
		if dotproduct < 0:
				return False

		squaredlengthba = (b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y)
		if dotproduct > squaredlengthba:
				return False

		return True

class Point:
		def __init__(self, x, y):
				self.x = x
				self.y = y
		
		def scale(self, s):
				self.x *= s
				self.y *= s
				return self

			
class RewardShaping:
		""" It takes care of everything just add in the environment loop and make
				sure that you reset it either by creating a new calling reset at the end
				of each episode otherwise last observation from prev episode is used which
				can raise some errors!
		"""
		def __init__(self, obs_prev = None, action_prev = None):
				self.reset(obs_prev, action_prev)
				
		def reset(self, obs_prev = None, action_prev = None):
				self.obs_prev = obs_prev
				self.action_prev = action_prev
				# for debug purpose
				self.obs_cur = None
				self.action_cur = None
				self.conseqActionCounter = 0
				self.dist2bombs_prev = 0
				self.notUsingAmmoCount = 0
				# catch enemy
				self.closestEnemyIdPrev = -1
				self.closestEnemyDistPrev = float("inf")
				
		def shape_it(self, obs_now, action_now):
				""" Shape the reward based on the current and previous observation
						input: current observation, current action, and received reward
						output: the shaped reward, sum of each criteria for the final reward
				"""
				self.obs_cur = obs_now
				self.action_cur = action_now
				if self.obs_prev is None:
						#print("first iteration")
						self.obs_prev = obs_now
						self.action_prev = action_now
						return 0, None
				#
				reward_tmp = {}
				# REWARD VALUES (NB: some of them used as factors not directly!)
				MOBILITY_RWRD = 0.1
#				 CONSEQ_ACT_RWRD = -0.5
				CONSEQ_ACT_RWRD = -0.0001
				
				PLNT_BOMB_NEAR_WOOD_RWRD = 0.05
				PLNT_BOMB_NEAR_ENEM_RWRD = 0.1
#				 ON_FLAMES_RWRD = -0.8
				ON_FLAMES_RWRD = -0.0001
				
				INCRS_DIST_WITH_BOMBS_RWRD = 0.05
				PICKED_POWER_RWRD = 0.1
#				 CATCH_ENEMY_RWRD = 0.1
				CATCH_ENEMY_RWRD = 0.001
				#
				# movement: + for mobility
				pose_t = np.array(obs_now['position']) # t
				pose_tm1 = np.array(self.obs_prev['position']) # t-1
				moveDist = np.linalg.norm(pose_t-pose_tm1)
				reward_tmp['mobility'] = MOBILITY_RWRD if moveDist > 0 else 0 # give reward
				#
				# consequative action: - for conseq actions
				if self.action_prev == action_now:
						self.conseqActionCounter += 1
				else:
						self.conseqActionCounter = 0
				if self.conseqActionCounter > 11:
						reward_tmp['conseqact'] = CONSEQ_ACT_RWRD
				# keeping ammo: - for not using its ammo
				if obs_now['ammo'] == self.obs_prev['ammo']:
						self.notUsingAmmoCount += 1
				else:
						self.notUsingAmmoCount = 0
				if self.notUsingAmmoCount > 11:
						reward_tmp['ammousage'] = CONSEQ_ACT_RWRD
				#
				# plant a bomb: + based on value of the bombing position
				bombs_pose = np.argwhere(obs_now['bomb_life'] != 0)
				if obs_now['ammo'] < self.obs_prev['ammo']:
						surroundings = [(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0)]
						mybomb_pose = self.obs_prev['position'] # equal to agent previous position
						# validate if the bomb actually exists there
						found_the_bomb = False
						for bp in bombs_pose:
								if np.equal(bp, mybomb_pose).all():
										found_the_bomb = True
										break
						assert found_the_bomb # end of validation
						nr_woods = 0
						nr_enemies = 0
						for p in surroundings:
								cell_pose = (mybomb_pose[0] + p[0], mybomb_pose[1] + p[1])
								if cell_pose[0] > 10 or cell_pose[1] > 10: # bigger than board size
										continue
								#print(obs_now['board'][cell_pose])
								nr_woods += obs_now['board'][cell_pose] == Item.Wood.value
								nr_enemies += obs_now['board'][cell_pose] in [e.value for e in obs_now['enemies']]
						#print("nr woods: ", nr_woods)
						#print("nr enemies: ", nr_enemies)
						assert nr_woods + nr_enemies < 10
						reward_tmp['plantbomb'] = \
								PLNT_BOMB_NEAR_WOOD_RWRD * nr_woods \
								+ PLNT_BOMB_NEAR_ENEM_RWRD * nr_enemies # give reward
				#
				# on Flames: - if agent on any blast direction 
				for bp in bombs_pose:
						def rot_deg90cw(point):
								new_point = [0, 0]
								new_point[0] = point[1]
								new_point[1] = -point[0]
								return new_point
						
						#print(type(bp))
						factor = 1/obs_now['bomb_life'][tuple(bp)] # inverse of time left
						blast_strength = obs_now['bomb_blast_strength'][tuple(bp)]

						# blast directions
						blast_N = Point(0,1).scale(blast_strength)
						blast_S = Point(0,-1).scale(blast_strength)
						blast_W = Point(-1,0).scale(blast_strength)
						blast_E = Point(1,0).scale(blast_strength)

						# agent on blast direction?
						bpPose = rot_deg90cw(bp)
						myPose = rot_deg90cw(obs_now['position'])
						myPose = Point(myPose[0]-bpPose[0], myPose[1]-bpPose[1]) # my pose relative to the bomb!
						onBlastDirect = isBetween(blast_N, blast_S, myPose) or \
														isBetween(blast_W, blast_E, myPose)
						if onBlastDirect:
								#print("time: ", obs_now['bomb_life'][tuple(bp)])
								#print("on blast: ", factor)
								reward_tmp['onflame'] = ON_FLAMES_RWRD * factor
				#
				# Bombs distance: + if total distance from bombs increased
				dist2bombs = 0
				for bp in bombs_pose:
						dist2bombs += np.linalg.norm(obs_now['position']-bp)
				dist_delta = dist2bombs - self.dist2bombs_prev
				self.dist2bombs_prev = dist2bombs
				#print(dist_delta)
				# TODO: this may not be good if the delta oscillates all the time
				if (dist_delta > 0 and moveDist):
						reward_tmp['bombsdistance'] = dist_delta * INCRS_DIST_WITH_BOMBS_RWRD
				# picked power: + for every picked power
				potentialPower = self.obs_prev['board'][obs_now['position']]
				picked_power = (potentialPower == Item.ExtraBomb.value) or\
											 (potentialPower == Item.IncrRange.value) or\
											 (potentialPower == Item.Kick.value)
				if picked_power:
						reward_tmp['pickedpower'] = PICKED_POWER_RWRD
				
				# catch enemy: + if closing distance with the nearest enemy
				def closestEnemy():
						myPose = obs_now['position']
						closestEnemyId = -1
						closestEnemyDist = float("inf")
						for e in obs_now['enemies']:
								enemyPose = np.argwhere(obs_now['board'] == e.value)
								if len(enemyPose) == 0:
										continue
								dist2Enemy = np.linalg.norm(myPose-enemyPose)
								if dist2Enemy <= closestEnemyDist:
										closestEnemyId = e.value
										closestEnemyDist = dist2Enemy
						return closestEnemyId, closestEnemyDist

				closestEnemyId_cur, closestEnemyDist_cur = closestEnemy()
				if self.closestEnemyIdPrev != closestEnemyId_cur:
						self.closestEnemyIdPrev = closestEnemyId_cur
						self.closestEnemyDistPrev = closestEnemyDist_cur
				else:
						CATCHING_TRHE = 4 # consider catching when close at most this much to the enemy
						if closestEnemyDist_cur < self.closestEnemyDistPrev and\
								closestEnemyDist_cur < CATCHING_TRHE:
								reward_tmp['catchenemy'] = CATCH_ENEMY_RWRD
								self.closestEnemyDistPrev = closestEnemyDist_cur
						if closestEnemyDist_cur <= 1.1: # got that close
								self.closestEnemyDistPrev = float("inf")
				#print("catching: ", closestEnemyIdPrev)
				
				# update previous obs and action
				self.obs_prev = obs_now
				self.action_prev = action_now
				
				# just a notice :
				for k,v in reward_tmp.items():
						if v >= 1.0 or v <= -1.0:
								print("reward for criteria '%s' is %f" % (k, v))
				
				# sum up rewards
				reward_shaped = sum(reward_tmp.values())
				return np.clip(reward_shaped, -0.9, 0.9), reward_tmp


################## SimpleAgent as reward (psudo-pretraining) ##################
class RewardShaping_Simple:
	def __init__(self):
		self.sa = SimpleAgent(config["agent"](agent_id, config["game_type"]))

	def reset(self):
		self.sa = SimpleAgent(config["agent"](agent_id, config["game_type"]))

	def shape_it(self, obs_now, action_now):
		action_simple = self.sa.act(obs_now, [])
		#off policy by using simple agent action
		if action_now == action_simple:
			return 0.001, action_simple
		else:
			return -0.001, action_simple



################## FEATURIZE ##################
#create one-hot encoded feature layers from obs data.

def new_featurize(obs):

	BOARD_SIZE = 11

	shape = (BOARD_SIZE, BOARD_SIZE, 1)

	def get_matrix(board, key):
		res = board[key]
		return res.reshape(shape).astype(np.float32)

	def get_map(board, item):
		map1 = np.zeros(shape)
		map1[board == item] = 1
		return map1

	def get_multi(board, item):
		map1 = np.zeros(shape)
		for i in item:
			map1[board == i] = 1
		return map1

#Makes bomb explosion pattern from location, strength and time. 
#Recursive due to possibility of a bomb blowing up other bombs before their time.
	def bombSpread(bombMap,orgMap,bomb_life):
		bombMap1 = np.zeros_like(bombMap)
		for x in range(11):
			for y in range(11):
				if (bomb_life[x][y] > 0): #bomb found
					strength = bombMap[x][y]
					time = bomb_life[x][y]
					bombMap1[x][y] = time
					for di in [(0,1),(1,0),(-1,0),(0,-1)]: # for each direction
						travel = 1
						while ((travel*di[0] + x <= 10) & (travel*di[0] + x >= 0) & #travel in-bounds, strength times
								(travel*di[1] + y <= 10) & (travel*di[1] + y >= 0) &
								(travel < strength)):
							#met other bomb, and timer is longer than current
							if (bomb_life[x+travel*di[0]][y+travel*di[1]] > time):
								# copy bomblife, update and rerun bombspread, return from here.
								bomb_life2 = np.copy(bomb_life)
								bomb_life2[x+travel*di[0]][y+travel*di[1]] = time
								return bombSpread(bombMap,orgMap,bomb_life2)
							bombMap1[x+travel*di[0]][y+travel*di[1]] = time
							if (orgMap[x+travel*di[0]][y+travel*di[1]]) in [1,2]: #unpassable
								break
							travel += 1
		return bombMap1 

### Centering (11 x 11 view)    
	def shift_board(arr, pos, fill_value=2):
		dx = 5 - pos[1]
		dy = 5 - pos[0]
		result = np.empty_like(arr)
		if dy > 0:
			result[:dy] = fill_value
			result[dy:] = arr[:-dy]
		elif dy < 0:
			result[dy:] = fill_value
			result[:dy] = arr[-dy:]
		else:
			result = arr
		arr= np.transpose(result)
		result = np.empty_like(arr)
		result = np.empty_like(arr)
		if dx > 0:
			result[:dx] = fill_value
			result[dx:] = arr[:-dx]
		elif dx < 0:
			result[dx:] = fill_value
			result[:dx] = arr[-dx:]
		else:
			result = arr
		result = np.transpose(result)
		return result    

	position = obs["position"]
	my_position = np.zeros(shape)
	my_position[position[0], position[1], 0] = 1    
    
	if CENTERED:
		obs['board'] 				= shift_board(obs['board'], position, fill_value=2)
		obs['bomb_blast_strength'] 	= shift_board(obs['bomb_blast_strength'], position, fill_value=0)
		obs['bomb_life'] 			= shift_board(obs['bomb_life'], position, fill_value=0)
		position = (5,5)# will always be this when centered    
    
    
	board = get_matrix(obs, 'board')
        
	path_map		= get_map(board, 0)					# Empty space
	rigid_map		= get_map(board, 1)					# Rigid = 1
	wood_map		= get_map(board, 2)					# Wood = 2
	flames_map		= get_map(board, 4)					# Flames = 4
	upgrade_map		= get_multi(board, [6,7,8])

	enemies = np.zeros(shape)
	for enemy in obs["enemies"]:
		enemies[board == enemy.value] = 1

	bomb_blast_strength = get_matrix(obs, 'bomb_blast_strength')
	bomb_life		    = get_matrix(obs, 'bomb_life')

	ammo			= np.full((BOARD_SIZE, BOARD_SIZE, 1), obs["ammo"])
	blast_strength  = np.full((BOARD_SIZE, BOARD_SIZE, 1), obs["blast_strength"])
	can_kick		= np.full((BOARD_SIZE, BOARD_SIZE, 1), int(obs["can_kick"]))

	maps = []
    
	if not CENTERED:
		maps = [my_position]

	if BLASTPAT:
    #blast pattern bomb map with count down.
		bomb_map = bombSpread(obs['bomb_blast_strength'],obs['board'],obs['bomb_life']) 
		bomb_map = bomb_map.reshape(shape).astype(np.float32)        
		maps += [bomb_map]
	else:
		bomb_map = get_map(board, 3)
		maps +=[bomb_map,bomb_blast_strength,bomb_life]     
        
	maps += [enemies,
			path_map,
			rigid_map,
			wood_map,
			flames_map,
			upgrade_map,
			ammo,
			blast_strength,
			can_kick]
	obs = np.concatenate(maps, axis=2)  

	return obs.astype(np.uint8)


################## Environment def ##################

class WrappedEnv(OpenAIGym):
	def __init__(self, gym, visualize=False):
		self.rewardShaping = RewardShaping()
		self.rewardShaping_Simple =RewardShaping_Simple()
		self.gym = gym
		self.visualize = visualize
		self.ecount = 0
		self.rewards = []
		self.take_actions = [0,0,0,0,0,0]

	def execute(self, action):
		if self.visualize:
			self.gym.render()

		obs = self.gym.get_observations() #old state

        #Simple agent reward shaping (Off policy)
		if (SAREWARD):
			agent_reward1, test_action = self.rewardShaping_Simple.shape_it(obs[self.gym.training_agent], action)
			action = test_action # Off policy, simple agent in control.
            
        #Pommerman x reward shaping
		if (PXREWARD):
			agent_reward1, _ = self.rewardShaping.shape_it(obs[self.gym.training_agent], action)            
        
		actions = self.unflatten_action(action=action)
		self.take_actions[actions] += 1

		all_actions = self.gym.act(obs)
		all_actions.insert(self.gym.training_agent, actions)
		state, reward, terminal, _ = self.gym.step(all_actions)
		agent_state = new_featurize(state[self.gym.training_agent]) #state changed by featurize
		agent_reward = reward[self.gym.training_agent]
				
		# End of episode:
		if (terminal):
				self.ecount += 1
				self.rewards.append(agent_reward)
                #random shuffle
				if (RANDOMSTART): 
					global agents
					global agent_id
					global env
					random.shuffle(agents)
					for idx,item in enumerate(agents):
						item.set_agent_id(idx)
						if(type(item) == TensorforceAgent):
							agent_id= idx
					env.set_training_agent(agents[agent_id].agent_id)
					env.set_agents(agents)
				ne = 10 # evaluate every ne episodes
                #Log env reward: (-1, 0, +1)
				if not self.ecount % ne:
						print("Finished episodes: {ep}; mean/median reward over last {ne} episodes: {meanr} / {medianr}; min reward:{minr}; max reward:{maxr}"
							.format(ep=self.ecount,
								ne = ne,
								meanr=np.mean(self.rewards[-ne:]),
								medianr=np.median(self.rewards[-ne:]),
								minr=np.min(self.rewards[-ne:]),
								maxr=np.max(self.rewards[-ne:])))
						with open('results/log_conv_ppo_lre3-ras.csv', 'a') as the_file2:
							writer = csv.writer(the_file2,delimiter = ',')
							line1 = [self.ecount,
									round(np.mean(self.rewards[-ne:]), 2),
									np.median(self.rewards[-ne:]),
									np.min(self.rewards[-ne:]),
									np.max(self.rewards[-ne:])]
							writer.writerow(line1)
				# REWARD shaping:
		if (PXREWARD or SAREWARD):
			agent_reward = agent_reward1                     
                    
		return agent_state, terminal, agent_reward

	def reset(self):
		obs = self.gym.reset()
		self.rewardShaping.reset()
		self.rewardShaping_Simple.reset()
		agent_obs = new_featurize(obs[agent_id])
		return agent_obs

################## Logging & Saving ##################
with open('results/log_conv_ppo_lre3-ras.csv', 'w') as the_file:
	header = ["episodes","mean timesteps", "mean reward", "median reward", "min reward", "max reward", "mc action"]
	writer = csv.writer(the_file,delimiter = ',')
	writer.writerow(header)
		
with open('results/log_conv_ppo_lre3-ras2.csv', 'w') as the_file2:
	header = ["episodes", "mean reward", "median reward", "min reward", "max reward"]
	writer = csv.writer(the_file2,delimiter = ',')
	writer.writerow(header)		 


# Callback function to print episode statistics
def episode_finished(r):
	ne = 10 # evaluate every ne episodes

	if not r.episode % ne:
		print("Finished episodes: {ep} mean timesteps: {ts}; mean/median reward over last episodes: {meanr} / {medianr}; min reward:{minr}; max reward:{maxr}; taken actions:{tac}"
			.format(ep=r.episode,
					ts=round(np.mean(r.episode_timesteps), 2),
					ne = ne,
					meanr=round(np.mean(r.episode_rewards[-ne:]), 2),
					medianr=round(np.median(r.episode_rewards[-ne:]), 2),
					minr=round(np.min(r.episode_rewards[-ne:]), 2),
					maxr=round(np.max(r.episode_rewards[-ne:]), 2),
					tac=wrapped_env.take_actions))

		with open('results/log_conv_ppo_lre3-ras.csv', 'a') as the_file:
			writer = csv.writer(the_file,delimiter = ',')
			line2 = [r.episode,
					np.mean(r.episode_timesteps[-ne:]),
					np.mean(r.episode_rewards[-ne:]),
					np.median(r.episode_rewards[-ne:]),
					np.min(r.episode_rewards[-ne:]),
					np.max(r.episode_rewards[-ne:]),
					wrapped_env.take_actions]
			writer.writerow(line2)

	if not r.episode % 1000:
		print("saving model")
		agent.save_model(directory="results/ppo_lre3-ras_model/")
	return True    
    

################## Run Environment ##################    
    
# Instantiate and the environment

wrapped_env = WrappedEnv(env, False) # set to false to disable rendering

#agent.restore_model(directory="/content/drive/My Drive/DeepLearning/Results/")

# Create the Tensorforce runner
# change agent here
runner = Runner(agent=agent, environment=wrapped_env)

# Start learning for n episodes
runner.run(episodes=EPISODES, max_episode_timesteps=2000, episode_finished=episode_finished)

try:
	runner.close()
except AttributeError as e:
	pass