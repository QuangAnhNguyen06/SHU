from controller import Supervisor
from controller import Node
from controller import Field
import math
import numpy as np
import random
import pickle




class MyCustomRobot(Supervisor):
    def __init__(self):

        super().__init__()
        self.MAX_SPEED = 6.28
        self.ANGLE_THRESHOLD = 1
        self.DISTANCE_THRESHOLD = 0.002
        self.DISTANCE = 0.25
        self.timestep = 32
        self.current_angle = 0

        self.goal_state = 28

        self.ps = []
        self.ps_names = [
            'ps0', 'ps1', 'ps2', 'ps3',
            'ps4', 'ps5', 'ps6', 'ps7'
        ]        


    def get_goal_state(self,goal_state):
        self.goal_state = goal_state

    def initialize_devices(self):
        # Sensors
        self.iu = self.getDevice('inertial unit')
        self.iu.enable(self.timestep)
        self.gps = self.getDevice('gps')
        self.gps.enable(self.timestep)

        for name in self.ps_names:
            device = self.getDevice(name)
            device.enable(self.timestep)
            self.ps.append(device)


        # Some devices, such as the InertialUnit, need some time to "warm up"
        self.wait()
        # Actuators
        self.leftMotor = self.getDevice('left wheel motor')
        self.rightMotor = self.getDevice('right wheel motor')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)

    def wait(self):
        """ Waits for ...ms """
        self.step(100)

    def get_yaw(self):

        values = self.iu.getRollPitchYaw()
        yaw = round(math.degrees(values[2]))
        # EAST 0°, NORTH 90°, WEST 180°, SOUTH -90°.
        if yaw < 0:
            yaw += 360
        return yaw

    def rotate_to(self, target_yaw):

        completed = False
        speed = 0.3
        # Are we rotating left or right?
        starting_yaw = self.get_yaw()
        
        # Calculate the difference between target and current angles
        angle_difference = target_yaw - starting_yaw
        # Ensure the angle difference is within the range [-180, 180]
        if angle_difference < -180:
            angle_difference += 360
        if angle_difference > 180:
            angle_difference -= 360
        # Determine the turn direction
        rotation_left = True if angle_difference > 0 else False
        
        while self.step(self.timestep) != -1:
            current_yaw = self.get_yaw()
            if abs(target_yaw - current_yaw) > self.ANGLE_THRESHOLD:
                if rotation_left:
                    leftSpeed = -speed * self.MAX_SPEED
                    rightSpeed = speed * self.MAX_SPEED
                else:
                    leftSpeed = speed * self.MAX_SPEED
                    rightSpeed = -speed * self.MAX_SPEED
            else:
                leftSpeed = 0.0
                rightSpeed = 0.0
                completed = True
            self.leftMotor.setVelocity(leftSpeed)
            self.rightMotor.setVelocity(rightSpeed)
            if completed:
                self.current_angle = target_yaw
                self.wait()
                return
                
    def turn_east(self):
        self.rotate_to(0)
        
    def turn_north(self):
        self.rotate_to(90)
        
    def turn_west(self):
        self.rotate_to(180)
        
    def turn_south(self):
        self.rotate_to(270)




    def get_state(self):
        position = self.gps.getValues()[0:2]
        # Get index of horizontal and vertical
        x_index = int((position[0] + 0.75) // 0.25)
        y_index = int((position[1] + 0.75) // 0.25)
    
        # Convert into state from 0 to 35
        state = y_index * 6 + x_index
    
        return state

    def get_ir_sensor(self):
        psValues = []
        for i in range(8):
            psValues.append(self.ps[i].getValue())
        return psValues
    

    def touch_object(self):
        psValues = self.get_ir_sensor()
        right_obstacle = psValues[0] > 120.0 or psValues[1] > 120.0 or psValues[2] > 120.0
        left_obstacle = psValues[5] > 120.0 or psValues[6] > 120.0 or psValues[7] > 120.0     
        if right_obstacle or left_obstacle:
            return True

        return False     
    
    # def is_stuck(self):
    #     start = self.gps.getValues()
    #     self.step(500)
    #     end = self.gps.getValues()
    #     if start == end:
    #         return True
    #     return False


    def stop_e_puck(self):
        self.leftMotor.setVelocity(0)
        self.rightMotor.setVelocity(0)        

    # Reset envi
    def reset_simulation(self):
        self.simulationReset() 
        #self.simulationResetPhysics() 


    # def started_node(self):
    #     robot_node = self.getFromDef("e-puck")   
    #     trans_field = self.getField(robot_node, "translation")
    #     rot_field = self.getField(robot_node, "rotation")



# 0 == east                     2
# 1 == west                  1     0
# 2 == north                    3
# 3 == south

    def do_action_east(self):
        robot.turn_east()
        robot.go_straight()

        is_touch =  self.touch_object()              
        a = get_state()
        if a == self.goal_state:
            return 20  
        if is_touch:
            return -20
        return -1
    
    def do_action_west(self):
        robot.turn_west()
        robot.go_straight()
    
        is_touch =  self.touch_object()              
        a = get_state()
        if a == self.goal_state:
            return 20  
        if is_touch:
            return -20
        return -1

    def do_action_north(self):
        robot.turn_north()
        robot.go_straight()

        is_touch =  self.touch_object()              
        a = get_state()
        if a == self.goal_state:
            return 20  
        if is_touch:
            return -20
        return -1

    def do_action_south(self):
        robot.turn_south()
        robot.go_straight()
    
        is_touch =  self.touch_object()              
        a = get_state()
        if a == self.goal_state:
            return 20  
        if is_touch:
            return -20
        return -1
    
    
    def go_straight(self):
        leftSpeed = self.MAX_SPEED
        rightSpeed = self.MAX_SPEED
        self.leftMotor.setVelocity(leftSpeed)
        self.rightMotor.setVelocity(rightSpeed)
        self.step(2000)
        self.leftMotor.setVelocity(0)
        self.rightMotor.setVelocity(0)       



#
# create q-table shape 36x4
#Q_TABLE = np.zeros((36, 4))
Q_TABLE = np.random.rand(36, 4)
random.seed(2206)
#epsilon = 0.2


def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        # Explore: Chọn một hành động ngẫu nhiên từ không gian hành động
        action = int(random.choice(range(4)))
    else:
        # Exploit- Khai thác: Chọn hành động tốt nhất từ Q-table cho trạng thái hiện tại
        action = max(range(len(Q[state])), key=lambda a: Q[state][a])
    return action
#action = epsilon_greedy(Q, state, epsilon)

    
def get_state():
    return robot.get_state()



# 0 == east                     2
# 1 == west                  1     0
# 2 == north                    3
# 3 == south


def get_reward(i):
    if i == 0:
        reward = robot.do_action_east()
    if i == 1:
        reward = robot.do_action_west()
    if i == 2:
        reward = robot.do_action_north()
    if i == 3:
        reward = robot.do_action_south()
    
    # if reward != 20 and reward != -100:
    #     reward = -1 
    #print(reward)
    return reward




#Q_TABLE
#alpha =
#gamma = 
def update_q_table(state, next_state, action, reward, alpha, gamma):
    """ Cập nhật giá trị của Q-table """
    state = int(get_state())
    # epsilon = 0.4
    action = epsilon_greedy(Q_TABLE,state,0.4)
    reward = get_reward(action)
    #if reward = -100:
    #if reward = 20:
    


    # action = action
    
    next_state = epsilon_greedy()

    old_q_value = Q_TABLE[state][action]
    max_q_value = max(Q_TABLE[next_state])
    new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * max_q_value)
    Q_TABLE[state][action] = new_q_value




# Q_TABLE = np.random.rand(36, 4)
def train(episode,alpha,gamma):
    for i in range(episode):
        #robot.reset_simulation()
        action = int(random.choice(range(4)))
        state = robot.get_state()
        terminate = False
        while not terminate:
            reward = get_reward(action)
            next_state = robot.get_state()
            next_action = epsilon_greedy(Q_TABLE,next_state,0.4)
            
            old_q_value = Q_TABLE[state][action]
            max_q_value = max(Q_TABLE[next_state])
            new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * max_q_value)
            Q_TABLE[state][action] = new_q_value        

            if reward == -20:
                terminate = True
            if reward == 20:
                terminate = True    

            state = next_state 
            action = next_action

        robot.reset_simulation()
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(Q_TABLE, f)
    return Q_TABLE




def load_q_table():
    with open('q_table.pkl', 'rb') as f:
        loaded_array = pickle.load(f)

    print(loaded_array)    

def load_q_table_test():
    with open('q_table_test.pkl', 'rb') as f:
        loaded_array = pickle.load(f)

    return(loaded_array)  
Q_TABLE_TEST = load_q_table_test()

def test(Q_TABLE):
    state = get_state()
    action = epsilon_greedy(Q_TABLE,state,0.1)
    get_reward(action)

    pass



threshold1 = 0.1
threshold2 = -10
def test_function(q_table, threshold1, threshold2, state):
    # Random số trong khoảng từ 0 đến 1
    random_num = np.random.rand()

    # Lấy hàng state từ bảng
    state_row = q_table[state]

    if random_num < threshold1:
        # Nếu số random nhỏ hơn threshold thứ 1, chọn random 1 giá trị từ hàng state
        chosen_value = np.random.choice(state_row)
        chosen_index = np.where(state_row == chosen_value)[0][0]
        return chosen_index
    elif random_num < threshold2:
        # Nếu số random nhỏ hơn threshold thứ 2, chọn lại
        return test_function(q_table, threshold1, threshold2, state)
    else:
        # Nếu số random lớn hơn threshold thứ 1, chọn số lớn nhất trong hàng state từ bảng
        max_value = np.max(state_row)
        max_index = np.where(state_row == max_value)[0][0]
        return max_index


# state_node = Node()
# def save_state():
#     state_node.saveState('state1')

# def load_state():
#     state_node.loadState('state1')



robot = MyCustomRobot()
robot.initialize_devices()
timestep = int(robot.getBasicTimeStep())
TIME_STEP = int(robot.getBasicTimeStep())


while robot.step(TIME_STEP) != -1:
    robot.step(20) 
    # robot.go_straight()
    # robot.stop_e_puck()
    
    #robot.started_node()

    # get_reward(2)
    # get_reward(2)
    # get_reward(2)
    # get_reward(2)
    # get_reward(2)
    # print(robot.get_ir_sensor())
    # print(robot.touch_object())
    # print(robot.get_state())
    # a = []
    # for i in range(10):
    #     a.append(epsilon_greedy(Q_TABLE,7, 0.5))



    # alpha = 0.2
    # gamma = 0.3
    # action = 2
    # state = robot.get_state()
    # terminate = False
    # while not terminate:
    #     reward = get_reward(action)
    #     next_state = robot.get_state()
    #     next_action = epsilon_greedy(Q_TABLE,next_state,0.4)
        
    #     old_q_value = Q_TABLE[state][action]
    #     max_q_value = max(Q_TABLE[next_state])
    #     new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * max_q_value)
    #     Q_TABLE[state][action] = new_q_value        
    #     print(reward)
    #     if reward == -20:
    #         terminate = True
    #     if reward == 20:
    #         terminate = True    
    #     else:
    #         terminate = True
    #     state = next_state 
    #     action = next_action

    # print(old_q_value)
    # print(new_q_value)

    # print(Q_TABLE)


    # train(1,0.2,0.2)
    # print(Q_TABLE*10)
    #load_q_table()

    # print(a)

    #robot.reset_simulation()
    #print(get_distance_to_goal(goal))
    #print(get_sensor_data())
    #reset_simulation()
    #print(get_state())
    # turn_west()
    # turn_east()
    # turn_north()
    # turn_south()
    #print(i)

    # robot.turn_west()
    # robot.move_forward()
    # print(robot.get_state())
    # robot.reset_simulation()

    # robot.turn_west()
    # robot.move_forward()
    # robot.move_forward()
    # if robot.touch_object() == True:
    #     print(robot.touch_object())
    #     robot.reset_simulation()

    while robot.get_state() != robot.goal_state:
        current_state = get_state()
        action = test_function(Q_TABLE_TEST,threshold1,threshold2,current_state)
        get_reward(action)
    #Q_TABLE_TEST
    if robot.get_state() == robot.goal_state:
        print('Success')
    
    break