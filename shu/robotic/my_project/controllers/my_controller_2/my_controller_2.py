from controller import Supervisor
import numpy as np

# Khởi tạo các tham số Q-Learning
learning_rate = 0.1
discount_factor = 0.95
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

# Định nghĩa môi trường
class EPuckRobot(Supervisor):
    def __init__(self):
        super(EPuckRobot, self).__init__()
        self.gps = self.getDevice("gps")
        self.gps.enable(int(self.getBasicTimeStep()))
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.target_position = np.array([0.25, -0.25, 0.01])
        self.state_space_size = 20  # Ví dụ về kích thước không gian trạng thái
        self.action_space_size = 5  # Định nghĩa số lượng hành động
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))

    def get_state(self):
        position = self.gps.getValues()  # Lấy vị trí hiện tại từ GPS
        if not all(np.isnan(position)):  # Kiểm tra xem tất cả giá trị có phải NaN không
        # Nếu không phải NaN, thực hiện tính toán trạng thái
            state = int((position[0] + 1) * 10)  # Chuyển đổi ví dụ
            return state
        else:
        # Xử lý trường hợp giá trị NaN, ví dụ: trả về một giá trị trạng thái mặc định hoặc log lỗi
            return None  # Hoặc một giá trị mặc định khác phù hợp với logic của bạn


    def step_action(self, action):
        # Thực hiện hành động bằng cách điều chỉnh tốc độ của motor, ví dụ:
        if action == 0:  # đi thẳng
            self.left_motor.setVelocity(5)
            self.right_motor.setVelocity(5)
        elif action == 1:  # quay phải
            self.left_motor.setVelocity(5)
            self.right_motor.setVelocity(-5)
        elif action == 2:  # quay trái
            self.left_motor.setVelocity(-5)
            self.right_motor.setVelocity(5)
        # Thêm các hành động khác nếu cần
        self.step(self.getBasicTimeStep())

    def get_reward(self, state):
        position = np.array(self.gps.getValues())
        distance_to_target = np.linalg.norm(position - self.target_position)
        reward = -distance_to_target
        return reward

    def update_q_table(self, state, action, reward, new_state):
        max_future_q = np.max(self.q_table[new_state])
        current_q = self.q_table[state, action]
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        self.q_table[state, action] = new_q

# Chạy mô phỏng và huấn luyện
robot = EPuckRobot()
num_episodes = 1000

for episode in range(num_episodes):
    state = robot.get_state()
    done = False

    while not done:
        # Lựa chọn hành động
        if np.random.uniform(0, 1) < exploration_rate:
            action = np.random.randint(0, robot.action_space_size)
        else:
            action = np.argmax(robot.q_table[state])

        # Thực hiện hành động và nhận phần thưởng
        robot.step_action(action)
        new_state = robot.get_state()
        reward = robot.get_reward(new_state)

        # Cập nhật Q-table
        robot.update_q_table(state, action, reward, new_state)

        state = new_state

        if reward > -0.01:  # Điều kiện kết thúc khi đạt gần mục tiêu
            done = True

    # Giảm exploration rate
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

print("Huấn luyện hoàn tất!")
