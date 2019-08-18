#### SELF PLAY
EPISODES = 5#30 # Number of self play before retraining the model
MCTS_SIMS = 12#50 # Number of MCTS simutations from one game state ( = number of node explored )
MEMORY_SIZE = 3000#3000#30000 # Number of memories (training sample) keep for the model retraining
TURNS_UNTIL_TAU0 = 10 # turn on which it starts playing deterministically
CPUCT = 1 # in MCTS, the constant determining the level of exploration
EPSILON = 0.2 # in MCTS, used if the exploration is at the root node. Lead to different formula of U term
ALPHA = 0.8


#### RETRAINING
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001 # Regulizer coefficient of the l2 regulizer of KERAS
LEARNING_RATE = 0.1 # Learning rate of the SGD
MOMENTUM = 0.9 # Momentum of the SGD
TRAINING_LOOPS = 5 # Number of time the networks is trained with a different randomly choosen training set (sampled from LT memories)
filters_num = 50#75
kernel_isze_num = 4
HIDDEN_CNN_LAYERS = [
	{'filters':filters_num, 'kernel_size': (kernel_isze_num,kernel_isze_num)}
	 , {'filters':filters_num, 'kernel_size': (kernel_isze_num,kernel_isze_num)}
	 , {'filters':filters_num, 'kernel_size': (kernel_isze_num,kernel_isze_num)}
	 , {'filters':filters_num, 'kernel_size': (kernel_isze_num,kernel_isze_num)}
	 , {'filters':filters_num, 'kernel_size': (kernel_isze_num,kernel_isze_num)}
	 , {'filters':filters_num, 'kernel_size': (kernel_isze_num,kernel_isze_num)}
	]

#### EVALUATION
EVAL_EPISODES = 5
SCORING_THRESHOLD = 1.3