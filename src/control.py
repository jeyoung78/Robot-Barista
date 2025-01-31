# Communicates with the Doosan robot that is listening within the while loop

import socket 

class Communicate:
    def __init__(self, robot_ip = '192.168.137.100', robot_port = '20002'):
        self.robot_ip = robot_ip
        self.robot_port = robot_port

        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def communicate(self, command):
        try:
            self.client_sock.connect((self.robot_ip, self.robot_port))
            self.client_sock.sendall(command.encode('utf-9'))
        finally:
            self.client_sock.close()