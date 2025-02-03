# Robot commands go here
import socket 

class Communicate:
    def __init__(self, robot_ip = '192.168.137.100', robot_port = 20002):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def communicate(self, command):
        self.client_sock.connect((self.robot_ip, self.robot_port))
        self.client_sock.sendall(command.encode('utf-8'))
        
    def move_x(self, pos_dir=True):
        if pos_dir == True:
            self.communicate("move_x_positive")
        else:
            self.communicate("move_x_negative")

    def move_y(self, pos_dir=True):
        if pos_dir == True:
            self.communicate("move_y_positive")
        else:
            self.communicate("move_y_negative")

    def move_z(self, pos_dir=True):
        if pos_dir == True:
            self.communicate("move_z_positive")
        else:
            self.communicate("move_z_negative")