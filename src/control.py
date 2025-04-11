import socket
import time

class Communicate:
    def __init__(self, robot_ip='192.168.137.100', robot_port=20002):
        self.robot_ip = robot_ip
        self.robot_port = robot_port

    def communicate(self, command):
        # Create a new socket for each command
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_sock:
            client_sock.connect((self.robot_ip, self.robot_port))
            client_sock.sendall(command.encode('utf-8'))

    def move_x(self, pos_dir=True):
        command = "move_x_positive" if pos_dir else "move_x_negative"
        self.communicate(command)

    def move_y(self, pos_dir=True):
        command = "move_y_positive" if pos_dir else "move_y_negative"
        self.communicate(command)

    def move_z(self, pos_dir=True):
        command = "move_z_positive" if pos_dir else "move_z_negative"
        self.communicate(command)

    def prepare(self, pos_dir=True):
        command = "prepare"
        self.communicate(command)

    def initial_pos(self):
        command = "initial" 
        self.communicate(command)

    def release(self):
        command = "release"
        self.communicate(command)

    def grab(self):
        command = "grab"
        self.communicate(command)

comm = Communicate()
comm.move_y(True)

def main():
    time.sleep(3)
    comm.grab()
    time.sleep(3)
    comm.release()
    time.sleep(3)
    comm.grab()
    time.sleep(3)
    comm.release()

if __name__ == "__main__":
    main()