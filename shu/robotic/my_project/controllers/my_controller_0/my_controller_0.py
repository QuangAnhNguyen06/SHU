"""my_controller_0 controller."""
from controller import Robot
from utilities import MyCustomRobot
from collections import deque
import numpy as np
from controller import Supervisor



# create the Robot instance.
#robot = Robot()
robot = MyCustomRobot(verbose=False)
robot.initialize_devices()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())




gps = robot.getDevice('gps')
gps.enable(timestep)
# Lấy cảm biến
ps = []
psNames = [
    'ps0', 'ps1', 'ps2', 'ps3',
    'ps4', 'ps5', 'ps6', 'ps7'
]
for i in range(8):
    ps.append(robot.getDevice(psNames[i]))
    ps[i].enable(timestep)


def move_up():
    robot.turn_north()
    robot.move_forward()    
def move_down():
    robot.turn_south()
    robot.move_forward()
def move_left():
    robot.turn_west()
    robot.move_forward()
def move_right():
    robot.turn_east()
    robot.move_forward()

def get_gps():
    gps_value = gps.getValues()[0:2]
    return gps_value


def get_state():
    position = get_gps()
    # Tính chỉ số ô dựa trên vị trí của vật thể
    x_index = int((position[0] + 0.75) // 0.25)
    y_index = int((position[1] + 0.75) // 0.25)
    
    # Chuyển đổi chỉ số ô thành chỉ số trạng thái
    state = y_index * 6 + x_index
    
    return state


Q = np.zeros((36,4)) # 36 state, 4 action
alpha = 0.1
gamma = 1
num_episodes = 1000





















n_states = 36  # Số lượng trạng thái
n_actions = 4  # Số lượng hành động
Q = np.zeros((n_states, n_actions))



# Chuyển đổi hành động thành hàm di chuyển
def move(state, action):
    if action == 0:
        move_up()
    elif action == 1:
        move_down()
    elif action == 2:
        move_left()
    elif action == 3:
        move_right()

    next_state = state  # Cập nhật dựa trên hành động thực tế
    reward = -1  # Phần thưởng mặc định
    if is_obstacle():
        reward = -100  # Phạt nặng nếu gặp chướng ngại vật
        done = True
    else:
        done = False

    return next_state, reward, done

# Các hàm khác như epsilon_greedy, update_Q_sarsa giữ nguyên
# Chính sách ε-greedy để chọn hành động
def epsilon_greedy(state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])

# Hàm cập nhật Q-value theo SARSA
def update_Q_sarsa(state, action, reward, next_state, next_action, alpha=0.1, gamma=0.95):
    predict = Q[state, action]
    target = reward + gamma * Q[next_state, next_action]
    Q[state, action] += alpha * (target - predict)


goal_state = 35
def is_goal(state):
    return state == goal_state


# Hàm chạy một episode
def run_episode():
    state = np.random.choice(n_states)  # Bắt đầu từ một trạng thái ngẫu nhiên
    action = epsilon_greedy(state)

    while True:
        next_state, reward, done = move(state, action)
        next_action = epsilon_greedy(next_state)

        update_Q_sarsa(state, action, reward, next_state, next_action)

        state, action = next_state, next_action
        
        if is_goal(next_state):
            reward = 100  # hoặc một giá trị thưởng phù hợp khác
            done = True

        if done:
            break

# Ví dụ chạy 100 episodes
# for _ in range(100):
#     run_episode()







# Reset môi trg
def reset_simulation():
    robot.simulationReset()
    robot.simulationResetPhysics()   


























while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    reset_simulation()

# Enter here exit cleanup code.







# def calculate_vector(from_position, to_position):
#     return [to_position[0] - from_position[0], to_position[2] - from_position[2]]

# def calculate_distance(vector):
#     return math.sqrt(vector[0]**2 + vector[1]**2)

# def calculate_angle(vector):
#     return math.atan2(vector[1], vector[0])



# supervisor = Supervisor()




# # Các biến môi trường
# TIME_STEP = int(supervisor.getBasicTimeStep())
# #TIME_STEP = 32
# MAX_SPEED = 6.28
# max_speed = 6.28
# goal = np.array([0.25, -0.25])
# floor_size = np.linalg.norm([1.5,1.5])
# # Số hành động có thể thực hiện
# action_space = 4




# # Lấy GPS và kích hoạt
# gps = supervisor.getDevice('gps')
# gps.enable(TIME_STEP)
# # # Lấy cảm biến
# # ps = []
# # psNames = [
# #     'ps0', 'ps1', 'ps2', 'ps3',
# #     'ps4', 'ps5', 'ps6', 'ps7'
# # ]
# # for i in range(8):
# #     ps.append(supervisor.getDevice(psNames[i]))
# #     ps[i].enable(TIME_STEP)

# # Lấy nút của vật thể mục tiêu
# #target_object_node = supervisor.getFromDef('goal')

# # Lấy motor và đặt vị trí
# left_motor = supervisor.getDevice('left wheel motor')
# right_motor = supervisor.getDevice('right wheel motor')
# left_motor.setPosition(float('inf'))
# right_motor.setPosition(float('inf'))
# left_motor.setVelocity(0.0)
# right_motor.setVelocity(0.0)


# # Lấy inertial unit
# iu = supervisor.getDevice('inertial unit')
# iu.enable(TIME_STEP)

















# def get_state():
#     position = gps.getValues()[0:2]
#     # Tính chỉ số ô dựa trên vị trí của vật thể
#     x_index = int((position[0] + 0.75) // 0.25)
#     y_index = int((position[1] + 0.75) // 0.25)
    
#     # Chuyển đổi chỉ số ô thành chỉ số trạng thái
#     state = y_index * 6 + x_index
    
#     return state


# # Reset môi trg
# def reset_simulation():
#     supervisor.simulationReset()
#     supervisor.simulationResetPhysics()        










# # min-max normalization
# def normalizer(value, min_value, max_value):

#     #Returns:
#     #- float: Normalized value

#     normalized_value = (value - min_value) / (max_value - min_value)        
#     return normalized_value



# def get_distance_to_goal(goal):
    
#     #Calculates and returns the normalized distance from the robot's current position to the goal.
    
#     #Returns:
#     #numpy.ndarray: Normalized distance vector.
    
    
#     gps_value = gps.getValues()[0:2]
#     current_coordinate = np.array(gps_value)
#     distance_to_goal = np.linalg.norm(goal - current_coordinate)
#     normalizied_coordinate_vector = normalizer(distance_to_goal, min_value=0, max_value=floor_size)
    
#     return normalizied_coordinate_vector




# # Lấy cảm biến
# ps = []
# psNames = [
#     'ps0', 'ps1', 'ps2', 'ps3',
#     'ps4', 'ps5', 'ps6', 'ps7'
# ]
# for i in range(8):
#     ps.append(supervisor.getDevice(psNames[i]))
#     ps[i].enable(TIME_STEP)


# def get_sensor_data():

#     psValues = []
#     for i in range(8):
#         psValues.append(ps[i].getValue())

        
#     #sensor_data = np.array(psValues)
#     sensor_data = np.array(psValues, dtype=np.float32)
#     #normalized_sensor_data = normalizer(sensor_data, min_sensor, max_sensor)
#     return sensor_data



# # concat 2 data: gps and distance sensor
# def get_observations():

    
#     normalized_sensor_data = get_sensor_data()
#     normalizied_current_coordinate = np.array([get_distance_to_goal()], dtype=np.float32)
    
#     state_vector = np.concatenate([normalizied_current_coordinate, normalized_sensor_data], dtype=np.float32)
#     return state_vector




# def get_yaw():
#     values = iu.getRollPitchYaw()
#     yaw = round(math.degrees(values[2]))
#     if yaw < 0:
#         yaw += 360
#     return yaw


# ANGLE_THRESHOLD = 1
# current_angle = 0

# def rotate_to(target_yaw,TIME_STEP = TIME_STEP):
#     """ Rotates the robot to one specific direction. """
#     completed = False
#     speed = 0.3
#     # Are we rotating left or right?
#     starting_yaw = get_yaw()
    
#     # Calculate the difference between target and current angles
#     angle_difference = target_yaw - starting_yaw
#     # Ensure the angle difference is within the range [-180, 180]
#     if angle_difference < -180:
#         angle_difference += 360
#     if angle_difference > 180:
#         angle_difference -= 360
#     # Determine the turn direction
#     rotation_left = True if angle_difference > 0 else False
    
#     while supervisor.step(TIME_STEP) != -1:
#         current_yaw = get_yaw()
#         if abs(target_yaw - current_yaw) > ANGLE_THRESHOLD:
#             if rotation_left:
#                 leftSpeed = -speed * MAX_SPEED
#                 rightSpeed = speed * MAX_SPEED
#             else:
#                 leftSpeed = speed * MAX_SPEED
#                 rightSpeed = -speed * MAX_SPEED
#         else:
#             leftSpeed = 0.0
#             rightSpeed = 0.0
#             completed = True
#         left_motor.setVelocity(leftSpeed)
#         right_motor.setVelocity(rightSpeed)
#         if completed:
#             current_angle = target_yaw
#             supervisor.step(500)
#             return 


# def turn_east():
#     current_angle = 0
#     rotate_to(0,TIME_STEP)
    
# def turn_north():
#     current_angle = 90
#     rotate_to(90,TIME_STEP)
    
# def turn_west():
#     current_angle = 180
#     rotate_to(180,TIME_STEP)
    
# def turn_south():
#     current_angle = 270
#     rotate_to(270,TIME_STEP)


# DISTANCE = 0.25
# DISTANCE_THRESHOLD = 0.002
# def move_forward():
#     """ Moves the robot forward for a set distance. """
#     speed = 0.5
#     starting_coordinate = gps.getValues()
    

#     # Calculate the desired ending coordinate
#     destination_coordinate = [
#         starting_coordinate[0] + DISTANCE * math.cos(math.radians(current_angle)),
#         starting_coordinate[1] + DISTANCE * math.sin(math.radians(current_angle))
#     ]
#     completed = False

#     print(current_angle) 
#     print(destination_coordinate)

#     while supervisor.step(TIME_STEP) != -1:
#         current_coordinate = gps.getValues()
#         distance_to_target_x = abs(current_coordinate[0] - destination_coordinate[0])
#         distance_to_target_y = abs(current_coordinate[1] - destination_coordinate[1])
#         print(distance_to_target_x)
#         print(distance_to_target_y)
#         print(current_coordinate)
        
#         if distance_to_target_x < DISTANCE_THRESHOLD and distance_to_target_y < DISTANCE_THRESHOLD:
#             leftSpeed = 0
#             rightSpeed = 0
#             completed = True
#         else:
#             leftSpeed = speed * MAX_SPEED
#             rightSpeed = speed * MAX_SPEED
#         left_motor.setVelocity(leftSpeed)
#         right_motor.setVelocity(rightSpeed)
#         if completed:
#             supervisor.step(500)
#             return


































# def apply_action(action):
#     left_motor.setPosition(float('inf'))
#     right_motor.setPosition(float('inf'))
    
#     if action == 0: # move forward
#         left_motor.setVelocity(max_speed)
#         right_motor.setVelocity(max_speed)
#     elif action == 1: # turn right
#         left_motor.setVelocity(max_speed)
#         right_motor.setVelocity(-max_speed)
#     elif action == 3: # turn left
#         left_motor.setVelocity(-max_speed)
#         right_motor.setVelocity(max_speed)
#     elif action == 2: # move backward
#         left_motor.setVelocity(-max_speed)
#         right_motor.setVelocity(-max_speed)
    

#     supervisor.step(500)
    

#     left_motor.setPosition(0)
#     right_motor.setPosition(0)
#     left_motor.setVelocity(0)
#     right_motor.setVelocity(0)      






# def step(action, max_steps):    
#     # Takes a step in the environment based on the given action.
#     # Returns:
#     # - state       = float numpy.ndarray with shape of (3,)
#     # - step_reward = float
#     # - done        = bool

    
#     apply_action(action)
#     step_reward, done = get_reward()
    
#     state = get_observations() # New state
    
#     # # Time-based termination condition
#     # if (int(getTime()) + 1) % max_steps == 0:
#     #     done = True
            
#     return state, step_reward, done




# reach_threshold = 0.003 # Distance threshold for considering the destination reached.

# def get_reward(self):
#     # Calculates and returns the reward based on the current state.
#     # Returns:
#     # - The reward and done flag.

#     done = False
#     reward = 0
    
#     normalized_sensor_data = get_sensor_data()
#     normalized_current_distance = get_distance_to_goal()
#     normalized_current_distance *= 100 # The value is between 0 and 1. Multiply by 100 will make the function work better
#     reach_threshold = reach_threshold * 100
    
#     # (1) Reward according to distance 
#     if normalized_current_distance < 42:
#         if normalized_current_distance < 10:
#             growth_factor = 5
#             A = 2.5
#         elif normalized_current_distance < 25:
#             growth_factor = 4
#             A = 1.5
#         elif normalized_current_distance < 37:
#             growth_factor = 2.5
#             A = 1.2
#         else:
#             growth_factor = 1.2
#             A = 0.9
#         reward += A * (1 - np.exp(-growth_factor * (1 / normalized_current_distance)))
        
#     else: 
#         reward += -normalized_current_distance / 100
        

#     # (2) Reward or punishment based on failure or completion of task
#     check_collision = self.touch.value
#     if normalized_current_distance < reach_threshold:
#         # Reward for finishing the task
#         done = True
#         reward += 25
#         print('+++ SOlVED +++')
#     elif check_collision:
#         # Punish if Collision
#         done = True
#         reward -= 5
        
        
#     # (3) Punish if close to obstacles
#     elif np.any(normalized_sensor_data[normalized_sensor_data > obstacle_threshold]):
#         reward -= 1

#     return reward, done










