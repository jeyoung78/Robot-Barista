# Goal now: move robot to grab a cup and move to a different location
# Monday: Implement robot code and Communicate class so that robot can be controlled from python script

from control import Communicate

def main():
    # url_image = 'images/IMG_2034.jpeg'
    # grab_cup(url_image)
    comm = Communicate()
    # comm.communicate("move_z_positive")
    comm.move_z(False)
    
def initial_alingment():
    # Make the end effector parallel to the table
    # Lower the robot to the cup level.
    # Make the end effector parallel to the x axis
    pass

def grab_cup(): 
    initial_alingment(url_image='images/IMG_2034.jpeg')
    # Note that x axis from the robot control perspective is y axis from the image
    # while y values of robot end effector and the cup align move the robot in x direction
    '''
    Image_stream.save_image_to_url()
    x_robot_ee, y_robot_ee = image_processing.find_red_dot()
    x_cup, y_cup = image_processing.find_cup()

    while y_robot_ee is around y_cup:
        move_robot_x()
        Image_stream.save_image_to_url()
        x_robot_ee, y_robot_ee = image_processing.find_red_dot()
        x_cup, y_cup = image_processing.find_cup()

    while x_robot_ee is around x_cup + buffer:
        move_robot_y()
        Image_stream.save_image_to_url()
        x_robot_ee, y_robot_ee = image_processing.find_red_dot()
        x_cup, y_cup = image_processing.find_cup()

    close_gripper()
    move_to_original_location()
    '''
    pass
    
if __name__ == "__main__":
    main() 