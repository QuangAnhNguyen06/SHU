"""gps_test controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import numpy as np
# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

gps = robot.getDevice('gps')
gps.enable(timestep)

ps = []
psNames = [
    'ps0', 'ps1', 'ps2', 'ps3',
    'ps4', 'ps5', 'ps6', 'ps7'
]
for i in range(8):
    ps.append(robot.getDevice(psNames[i]))
    ps[i].enable(timestep)

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


# Main loop
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # psValues = []
    # for i in range(8):
        # psValues.append(ps[i].getValue())
    # sensor_data = np.array(psValues, dtype=np.float32)    
    # print(sensor_data)
    
    # gps_value = gps.getValues()[0:2]
    # print(gps_value)
    
    print(get_state())
    
    
    
    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    

# Enter here exit cleanup code.
