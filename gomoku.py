#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:38:58 2019

@author: kruszylo
"""

import numpy as np
import imutils
from PIL import ImageGrab
from PIL import ImageDraw, Image
import cv2
import time

import numpy as np
np.set_printoptions(suppress=True)

from shutil import copyfile
import random
from importlib import reload


from keras.utils import plot_model

from game import Game, GameState
from agent import Agent
from memory import Memory
from model import Residual_CNN
from funcs import playMatches, playMatchesBetweenVersions

import loggers as lg

from settings import run_folder, run_archive_folder
import initialise
import pickle
from agent import User
from game import *
EMPTY_FIELD = 0
WHITE_TOKEN = 1
BLACK_TOKEN = -1
BOARD_PADDING = 27*2
BOARD_SIZE = 15 # 15 x 15
CELL_SIZE = 37*2 #37 x 37
FRAME_SIZE = BOARD_SIZE * CELL_SIZE + BOARD_PADDING*2
WINDOW_SIZE = 300
#first token position must be (25*2,25*2) 
DIAMETER_OF_TOKEN = CELL_SIZE #some default value
LEFT_TOP_POINT = (25,245)
RIGHT_BOTTOM_POINT = (LEFT_TOP_POINT[0]+FRAME_SIZE, LEFT_TOP_POINT[1]+FRAME_SIZE)
SCREEN_BOX = [LEFT_TOP_POINT[0],LEFT_TOP_POINT[1],RIGHT_BOTTOM_POINT[0],RIGHT_BOTTOM_POINT[1]]

def init_board():
    board = [[EMPTY_FIELD for i in range(BOARD_SIZE)] for j in range(BOARD_SIZE)]
    return board

def update_board(board, token_centers, token_color):
    for i,(x,y) in enumerate(token_centers):
        board_i = min((y - BOARD_PADDING)//CELL_SIZE, BOARD_SIZE - 1)
        board_j = min((x - BOARD_PADDING)//CELL_SIZE, BOARD_SIZE - 1)
        board[board_i][board_j] = token_color
    return board
    

def remove_lines(gray,replace_color):
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength=100
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=160)
    
    a,b,c = lines.shape
    for i in range(a):
        x = lines[i][0][0] - lines [i][0][2]
        y = lines[i][0][1] - lines [i][0][3]
        if x!= 0:
            if abs(y/x) <1:
                cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (int(replace_color[0]),int(replace_color[1]),int(replace_color[2])), 1, cv2.LINE_AA)
        if y!=0:
            if abs(x/y) <1:
                cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (int(replace_color[0]),int(replace_color[1]),int(replace_color[2])), 1, cv2.LINE_AA)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (3,3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
    return gray

def draw_green_circles(img, draw_on_img, typ):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    return gray,[]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
#    return binary,[]
    sx = cv2.Sobel(binary,cv2.CV_32F,1,0)
    sy = cv2.Sobel(binary,cv2.CV_32F,0,1)
    m = cv2.magnitude(sx,sy)
    binary = cv2.normalize(m,None,0.,255.,cv2.NORM_MINMAX,cv2.CV_8U)
#    return binary,[]
    #--- find contours ---
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    #--- select contours having a parent contour and append them to a list ---
    l = []
    for h in hierarchy[0]:
#        if h[0] > -1 and h[2] > -1:
        l.append(h[2])
    centers = []
    diameters = []
    #--- draw those contours ---
    for cnt in l:
        if cnt > 0:
            area = cv2.contourArea(contours[cnt])
            #normaly token has area about 4300
            if area > 2000:
                m = cv2.moments(contours[cnt])
                br = cv2.boundingRect(contours[cnt])
                diameters.append(br[2])
                if m['m00'] != 0:
                    #calc centroid
                    center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
                    centers.append(center)
                else:
                    print('WARNING zero division!')
                    cx = 0
                    cy = 0
                    for p in contours[cnt]:
                        cx += p[0][0]
                        cy += p[0][1]
                    cx = int(cx/len(contours[cnt]))
                    cy = int(cy/len(contours[cnt]))
                    centers.append((cx,cy))
                cv2.drawContours(draw_on_img, [contours[cnt]], 0, (0,255,0), 2)
#    global DIAMETER_OF_TOKEN
#    if len(diameters) > 0 : DIAMETER_OF_TOKEN = diameters[0]
#    print(f'There are {len(centers)} {typ} circles')
#    for center,diameter in zip(centers,diameters):
#        #cv2.circle(drawing, center, 3, (255, 0, 0), -1)
#        cv2.circle(draw_on_img, center, int(diameter/2)*2, (255, 0, 0), 1)
    return draw_on_img,centers

def drawGreen_circels(img, board):
    replace_color = img[10,10]#(numpy_img[3,3][0],numpy_img[3,3][1],numpy_img[3,3][2])
    draw_on_img = img.copy()
    img = remove_lines(img,replace_color)
    
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    white_tokens = cv2.dilate(img, el, iterations=6)
    white_tokens = cv2.bitwise_not(white_tokens)
    
    hsv = cv2.cvtColor(white_tokens, cv2.COLOR_BGR2HSV)
    value = 42 #whatever value you want to add
    hsv += value
    white_tokens = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    black_tokens = cv2.bitwise_not(img)
    
    black_tokens = cv2.dilate(black_tokens, el, iterations=6)
    
    black_tokens = cv2.bitwise_not(black_tokens)
#    return black_tokens,board
    draw_on_img,white_tokens_centers = draw_green_circles(white_tokens,draw_on_img, 'white')
    
    draw_on_img,black_tokens_centers = draw_green_circles(black_tokens, draw_on_img, 'black')
#    return draw_on_img,board
    board = init_board()
    board = update_board(board, white_tokens_centers, WHITE_TOKEN)
    board = update_board(board, black_tokens_centers, BLACK_TOKEN)
    #rectangle to help fitting board
    #draw_on_img = cv2.rectangle(draw_on_img, (BOARD_PADDING, BOARD_PADDING), (BOARD_PADDING+BOARD_SIZE * CELL_SIZE, BOARD_PADDING+BOARD_SIZE * CELL_SIZE), (0,0,255), 2)
    return draw_on_img, board

def draw_action(img, center):
#    DIAMETER_OF_TOKEN = 50
    return cv2.circle(img, center, int(DIAMETER_OF_TOKEN/2), (0, 0, 255), 2)
def display_board(board):
    for row in board:
        for field in row:
            if field == EMPTY_FIELD:
                print(' + ', end='')
            elif field == WHITE_TOKEN:
                print(' X ', end='')
            elif field == BLACK_TOKEN:
                print(' 0 ', end='')
            else:
                print(' ? ', end='')
        print()
    print()

def count_tokens(board, token_type):
    counter = 0
    for row in board:
        for field in row:
            if field == token_type:
                counter+=1
    return counter
def who_starts(board, player1, player2):
    black_count = 0
    for row in board:
        for field in board:
            if field == BLACK_TOKEN:
                black_count+=1
    if black_count == 0:
       print(player2.name + ' plays as X')
       state = GameState(np.array(board).flatten(), 1)
       action, pi, MCTS_value, NN_value = player2.act(state, 1)
       return action
    elif black_count == 1:
       print(player1.name + ' plays as X')
       print('CALL ME IF YOU WILL NEED HELP, PRESS space')
       global player2_tokens
       player2_tokens = WHITE_TOKEN
       return 1
env = Game()   
# If loading an existing neural network, copy the config file to root
if initialise.INITIAL_RUN_NUMBER != None:
    copyfile(run_archive_folder + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + '/config.py', './config.py')

import config

######## LOAD MEMORIES IF NECESSARY ########

if initialise.INITIAL_MEMORY_VERSION == None:
    memory = Memory(config.MEMORY_SIZE)
else:
    print('LOADING MEMORY VERSION ' + str(initialise.INITIAL_MEMORY_VERSION) + '...')
    memory = pickle.load( open( run_archive_folder + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + "/memory/memory" + str(initialise.INITIAL_MEMORY_VERSION).zfill(4) + ".p",   "rb" ) )
    
best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) +  env.grid_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
#If loading an existing neural netwrok, set the weights from that model
if initialise.INITIAL_MODEL_VERSION != None:
    best_player_version  = initialise.INITIAL_MODEL_VERSION
    print('LOADING MODEL VERSION ' + str(initialise.INITIAL_MODEL_VERSION) + '...')
    m_tmp = best_NN.read(env.name, initialise.INITIAL_RUN_NUMBER, best_player_version)

    best_NN.model.set_weights(m_tmp.get_weights())
#otherwise just ensure the weights on the two players are the same
else:
    best_player_version = 0

print('\n')
last_time = time.time()
winname = "AI_VISION"
cv2.namedWindow(winname, cv2.WINDOW_NORMAL)        # Create a named window
cv2.resizeWindow(winname, WINDOW_SIZE,WINDOW_SIZE)
cv2.moveWindow(winname, 950,50)  # Move window right corner of screen
board = init_board()

player1 = User('player1', env.state_size, env.action_size)
#print('creating AI')
player2 = Agent('best_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)
player2_tokens = BLACK_TOKEN
action_i, action_j, show_action, player2_before_act_tokens_count = -1, -1, False, 0
#display_board(board)
print('AI ready to play')
while(True):
    screen = ImageGrab.grab(bbox=(SCREEN_BOX))
    numpy_screen = np.array(screen)
    new_screen,board = drawGreen_circels(numpy_screen, board)
    if(show_action):
#        print((int(action_i*CELL_SIZE+BOARD_PADDING), int(action_j*CELL_SIZE+BOARD_PADDING)))
        new_screen = draw_action(new_screen, (int(action_i*(CELL_SIZE+3)+BOARD_PADDING), int(action_j*(CELL_SIZE+3)+BOARD_PADDING))) #TODO: +3 just to fit board
        current_tokens_count = count_tokens(board, player2_tokens)
        if(current_tokens_count > player2_before_act_tokens_count):
            show_action = False
    display_screen = cv2.resize(new_screen, (250,250))
    
    cv2.imshow(winname, display_screen)
#    print('Loop took {} seconds'.format(time.time() - last_time))
    #cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
#    k = cv2.waitKey(33)
#    if k==27:    # Esc key to stop
#        break
#    elif k==-1:  # normally -1 returned,so don't print it
#        continue
#    else:1
#        print(k) # else print its value
    key = cv2.waitKey(10)
    if key == ord(' '):
        print('(space pressed):MAKING DECISION...')
        state = GameState(np.array(board).flatten(), player2_tokens)
        action, pi, MCTS_value, NN_value = player2.act(state, 1)
        action_j, action_i = action//BOARD_SIZE, action - (action//BOARD_SIZE)*BOARD_SIZE
        print(f'PUT YOUR TOKEN ON POSITION y:{action_i}, x:{action_j}')
        player2_before_act_tokens_count = count_tokens(board, player2_tokens)
        show_action = True
    elif key == ord('s'):
        print('(s pressed):STARTING GAME. UNDERSTANDING WHO STARTS...')
        action = who_starts(board, player1, player2)
        #action = np.random.choice([0,1,2,14])
        action_j, action_i = action,action#action//BOARD_SIZE, action - (action//BOARD_SIZE)*BOARD_SIZE
        print(f'PUT YOUR TOKEN ON POSITION y:{action_i}, x:{action_j}')
        player2_before_act_tokens_count = count_tokens(board, player2_tokens)
        show_action = True
    elif key == ord('p'):
        print('(p pressed)')
        display_board(board)
    elif key & 0xFF == ord('q'):
        print('(q pressed): CLOSING WINDOW. BYE.')
        display_board(board)
        cv2.destroyAllWindows()
        break