


import numpy as np
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *



class Chess_Env:
    
    def __init__(self,N_grid):
        
        
        self.N_grid=N_grid                     # SIZE OF THE BOARD 棋盘大小
        
        self.Board=np.zeros([N_grid,N_grid])   # THE BOARD, THIS WILL BE FILLED BY 0 (NO PIECE), 1 (AGENT'S KING), 2 (AGENT'S QUEEN), 3 (OPPONENT'S KING) 棋盘，这将由0（没有棋子）、1（代理人的国王）、2（代理人的皇后）、3（对手的国王）组成。
        
        self.p_k1=np.zeros([2,1])              # POSITION OF THE AGENT'S KING AS COORDINATES 代理人的国王的位置作为坐标 * 【2行1列】
        self.p_k2=np.zeros([2,1])              # POSITION OF THE OPPOENT'S KING AS COORDINATES 对方国王的位置为坐标
        self.p_q1=np.zeros([2,1])              # POSITION OF THE AGENT'S QUEEN AS COORDINATES 代理人的女王的位置作为坐标
        
        self.dfk1=np.zeros([N_grid,N_grid])    # ALL POSSIBLE ACTIONS FOR THE AGENT'S KING (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES) 代理人的国王的所有可能行动（在没有其他棋子存在的情况下，国王可以移动的位置） * 【棋盘大小*棋盘大小】
        self.dfk2=np.zeros([N_grid,N_grid])    # ALL POSSIBLE ACTIONS FOR THE OPPONENT'S KING (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES) 对方国王的所有可能行动（在没有其他棋子存在的情况下，国王可以移动的位置）
        self.dfq1=np.zeros([N_grid,N_grid])    # ALL POSSIBLE ACTIONS FOR THE AGENT'S QUEEN (LOCATIONS WHERE IT CAN MOVE WITHOUT THE PRESENCE OF THE OTHER PIECES)
        
        self.dfk1_constrain=np.zeros([N_grid,N_grid])  # ALLOWED ACTIONS FOR THE AGENT'S KING CONSIDERING ALSO THE OTHER PIECES 考虑到其他棋子的情况，允许代理人的国王采取的行动。【constrain】
        self.dfk2_constrain=np.zeros([N_grid,N_grid])  # ALLOWED ACTIONS FOT THE OPPONENT'S KING CONSIDERING ALSO THE OTHER PIECES
        self.dfq1_constrain=np.zeros([N_grid,N_grid])  # ALLOWED ACTIONS FOT THE AGENT'S QUEEN CONSIDERING ALSO THE OTHER PIECES
        
        self.ak1=np.zeros([8])                         # ALLOWED ACTIONS OF THE AGENT'S KING (CONSIDERING OTHER PIECES), ONE-HOT ENCODED 代理人的国王允许的行动（考虑到其他棋子），one-hot编码
        self.possible_king_a=np.shape(self.ak1)[0]     # TOTAL NUMBER OF POSSIBLE ACTIONS FOR AGENT'S KING 代理人王的可能行动总数 【np.shape(ak1) = (8,)】
        
        self.aq1=np.zeros([8*(self.N_grid-1)])         # ALLOWED ACTIONS OF THE AGENT'S QUEEN (CONSIDERING OTHER PIECES), ONE-HOT ENCODED
        self.possible_queen_a=np.shape(self.aq1)[0]     # TOTAL NUMBER OF POSSIBLE ACTIONS FOR AGENT'S QUEEN 代理人的女王可能采取的行动总数

        self.check=0                                   # 1 (0) IF ENEMY KING (NOT) IN CHECK 1 (0) 如果敌方国王(不)受制于人

        # THIS MAP IS USEFUL FOR US TO UNDERSTAND THE DIRECTION OF MOVEMENT GIVEN THE ACTION MADE (SKIP...) 这个map对我们了解所采取的行动的运动方向很有用（跳过......）【共八个方向的跳跃移动的方式】
        self.map=np.array([[1, 0],
                            [-1, 0],
                            [0, 1],
                            [0, -1],
                            [1, 1],
                            [1, -1],
                            [-1, 1],
                            [-1, -1]])

        
        
    def Initialise_game(self):
        
        
        # START THE GAME BY SETTING PIECIES 通过设置棋子开始游戏
        
        self.Board,self.p_k2,self.p_k1,self.p_q1=generate_game(self.N_grid) # 生成棋盘之后的4x4数组；对方king的坐标; 己方King 和 Queen的坐标
       
        # Allowed actions for the agent's king 代理人的国王 允许的行动
        self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(self.p_k1, self.p_k2, self.p_q1, self.Board)
        
        # Allowed actions for the agent's queen
        self.dfq1_constrain, self.a_q1, self.dfq1  = degree_freedom_queen(self.p_k1, self.p_k2, self.p_q1, self.Board)
        
        # Allowed actions for the enemy's king
        self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(self.dfk1, self.p_k2, self.dfq1, self.Board, self.p_k1)
        
        # ALLOWED ACTIONS FOR THE AGENT, ONE-HOT ENCODED
        allowed_a=np.concatenate([self.a_q1,self.a_k1],0)
        
        # FEATURES (INPUT TO NN) AT THIS POSITION
        X=self.Features()
        
        
        
        
        return self.Board, X, allowed_a
        
    
    def OneStep(self,a_agent):
        
        # SET REWARD TO ZERO IF GAME IS NOT ENDED 如果游戏没有结束，将奖励设为零
        R=0
        # SET Done TO ZERO (GAME NOT ENDED) 将 "完成 "设置为 "零"（游戏未结束）。
        Done=0
        
        
        # PERFORM THE AGENT'S ACTION ON THE CHESS BOARD 在棋盘上执行代理人的行动
        
        if a_agent < self.possible_queen_a:    # THE AGENT MOVED ITS QUEEN 
           
           # UPDATE QUEEN'S POSITION
           direction = int(np.ceil((a_agent + 1) / (self.N_grid - 1))) - 1
           steps = a_agent - direction * (self.N_grid - 1) + 1

           self.Board[self.p_q1[0], self.p_q1[1]] = 0
           
           mov = self.map[direction, :] * steps
           self.Board[self.p_q1[0] + mov[0], self.p_q1[1] + mov[1]] = 2
           self.p_q1[0] = self.p_q1[0] + mov[0]
           self.p_q1[1] = self.p_q1[1] + mov[1]

        else:                                 # THE AGENT MOVED ITS KING                               
           
           # UPDATE KING'S POSITION
           direction = a_agent - self.possible_queen_a
           steps = 1

           self.Board[self.p_k1[0], self.p_k1[1]] = 0
           mov = self.map[direction, :] * steps
           self.Board[self.p_k1[0] + mov[0], self.p_k1[1] + mov[1]] = 1
           self.p_k1[0] = self.p_k1[0] + mov[0]
           self.p_k1[1] = self.p_k1[1] + mov[1]

        
        # COMPUTE THE ALLOWED ACTIONS AFTER AGENT'S ACTION
        # Allowed actions for the agent's king
        self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(self.p_k1, self.p_k2, self.p_q1, self.Board)
        
        # Allowed actions for the agent's queen
        self.dfq1_constrain, self.a_q1, self.dfq1  = degree_freedom_queen(self.p_k1, self.p_k2, self.p_q1, self.Board)
        
        # Allowed actions for the enemy's king
        self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(self.dfk1, self.p_k2, self.dfq1, self.Board, self.p_k1)

        
        # CHECK IF POSITION IS A CHECMATE, DRAW, OR THE GAME CONTINUES
        
        # CASE OF CHECKMATE
        if np.sum(self.dfk2_constrain) == 0 and self.dfq1[self.p_k2[0], self.p_k2[1]] == 1:
           
            # King 2 has no freedom and it is checked
            # Checkmate and collect reward
            Done = 1       # The epsiode ends
            R = 1          # Reward for checkmate
            allowed_a=[]   # Allowed_a set to nothing (end of the episode)
            X=[]           # Features set to nothing (end of the episode)
        
        # CASE OF DRAW
        elif np.sum(self.dfk2_constrain) == 0 and self.dfq1[self.p_k2[0], self.p_k2[1]] == 0:
           
            # King 2 has no freedom but it is not checked
            Done = 1        # The epsiode ends
            R = 0.       # Reward for draw
            allowed_a=[]    # Allowed_a set to nothing (end of the episode)
            X=[]            # Features set to nothing (end of the episode)
        
        # THE GAME CONTINUES
        else:
            
            # THE OPPONENT MOVES THE KING IN A RANDOM SAFE LOCATION
            allowed_enemy_a = np.where(self.a_k2 > 0)[0]
            a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
            a_enemy = allowed_enemy_a[a_help]

            direction = a_enemy
            steps = 1

            self.Board[self.p_k2[0], self.p_k2[1]] = 0
            mov = self.map[direction, :] * steps
            self.Board[self.p_k2[0] + mov[0], self.p_k2[1] + mov[1]] = 3

            self.p_k2[0] = self.p_k2[0] + mov[0]
            self.p_k2[1] = self.p_k2[1] + mov[1]
            
            
            
            # COMPUTE THE ALLOWED ACTIONS AFTER THE OPPONENT'S ACTION
            # Possible actions of the King
            self.dfk1_constrain, self.a_k1, self.dfk1 = degree_freedom_king1(self.p_k1, self.p_k2, self.p_q1, self.Board)
            
            # Allowed actions for the agent's king
            self.dfq1_constrain, self.a_q1, self.dfq1  = degree_freedom_queen(self.p_k1, self.p_k2, self.p_q1, self.Board)
            
            # Allowed actions for the enemy's king
            self.dfk2_constrain, self.a_k2, self.check = degree_freedom_king2(self.dfk1, self.p_k2, self.dfq1, self.Board, self.p_k1)

            # ALLOWED ACTIONS FOR THE AGENT, ONE-HOT ENCODED
            allowed_a=np.concatenate([self.a_q1,self.a_k1],0)
            # FEATURES
            X=self.Features()
            
            
        
        return self.Board, X, allowed_a, R, Done
        
        
    # DEFINITION OF THE FEATURES (SEE ALSO ASSIGNMENT DESCRIPTION)
    def Features(self):
        
        
        s_k1 = np.array(self.Board == 1).astype(float).reshape(-1)   # FEATURES FOR KING POSITION 找到king在棋盘上的位置，标记为1，其余皆为0，然后压缩成1维数组
        s_q1 = np.array(self.Board == 2).astype(float).reshape(-1)   # FEATURES FOR QUEEN POSITION
        s_k2 = np.array(self.Board == 3).astype(float).reshape(-1)   # FEATURE FOR ENEMY'S KING POSITION
        
        check=np.zeros([2])    # CHECK? FEATURE
        check[self.check]=1   # one-hot 编码 [1,0] not check; [0,1] checked
        
        K2dof=np.zeros([8])   # NUMBER OF ALLOWED ACTIONS FOR ENEMY'S KING, ONE-HOT ENCODED 敌方国王允许的行动数量，one-hot
        K2dof[np.sum(self.dfk2_constrain).astype(int)]=1
        
        # ALL FEATURES...
        x = np.concatenate([s_k1, s_q1, s_k2, check, K2dof],0) # 拼接在一起，包括 [己方 King 的位置] [己方 Queen 的位置] [对方 King 的位置] [是否 check] [敌方国王允许的行动数量]
        
        return x
        
        


        
        
        
        
        
        
