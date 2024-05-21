from controller import Robot
from controller import Supervisor
import math

class MyCustomRobot(Robot):
    def __init__(self, verbose=True):
        """ Initializes the robot and the required devices. """
        super().__init__()
        self.MAX_SPEED = 6.28
        self.ANGLE_THRESHOLD = 1
        self.DISTANCE_THRESHOLD = 0.002
        self.DISTANCE = 0.1
        self.timestep = 32
        self.verbose = verbose
        self.current_angle = 0
        
    def initialize_devices(self):
        """ Initializes sensors and actuators. """
        # Sensors
        self.iu = self.getDevice('inertial unit')
        self.iu.enable(self.timestep)
        self.gps = self.getDevice('gps')
        self.gps.enable(self.timestep)
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
        """ Waits for 500ms. """
        self.step(500)
        
    def get_yaw(self):
        """ Gets the yaw value from the InertialSensor, 
        converts it in degrees and normalizes it. """
        values = self.iu.getRollPitchYaw()
        yaw = round(math.degrees(values[2]))
        # The InertialUnit gives us a reading based on how the robot is oriented with
        # respect to the X axis of the world: EAST 0°, NORTH 90°, WEST 180°, SOUTH -90°.
        # This operation converts these values to a normal, positive circumfrence.
        if yaw < 0:
            yaw += 360
        return yaw
        
    def rotate_to(self, target_yaw):
        """ Rotates the robot to one specific direction. """
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
        if self.verbose:
            print("Rotating EAST")
        self.rotate_to(0)
        
    def turn_north(self):
        if self.verbose:
            print("Rotating NORTH")
        self.rotate_to(90)
        
    def turn_west(self):
        if self.verbose:
            print("Rotating WEST")
        self.rotate_to(180)
        
    def turn_south(self):
        if self.verbose:
            print("Rotating SOUTH")
        self.rotate_to(270)
        
    def move_forward(self):
        """ Moves the robot forward for a set distance. """
        if self.verbose:
            print("Moving forward")
        speed = 0.5
        starting_coordinate = self.gps.getValues()
        # Calculate the desired ending coordinate
        destination_coordinate = [
            starting_coordinate[0] + self.DISTANCE * math.cos(math.radians(self.current_angle)),
            starting_coordinate[1] + self.DISTANCE * math.sin(math.radians(self.current_angle))
        ]
        completed = False
        while self.step(self.timestep) != -1:
            current_coordinate = self.gps.getValues()
            distance_to_target_x = abs(current_coordinate[0] - destination_coordinate[0])
            distance_to_target_y = abs(current_coordinate[1] - destination_coordinate[1])
            if distance_to_target_x < self.DISTANCE_THRESHOLD and distance_to_target_y < self.DISTANCE_THRESHOLD:
                leftSpeed = 0
                rightSpeed = 0
                completed = True
            else:
                leftSpeed = speed * self.MAX_SPEED
                rightSpeed = speed * self.MAX_SPEED
            self.leftMotor.setVelocity(leftSpeed)
            self.rightMotor.setVelocity(rightSpeed)
            if completed:
                self.wait()
                return
