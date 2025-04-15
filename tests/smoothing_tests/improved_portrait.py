from EnhancedRobotClass import EnhancedRobotClass

# Create robot instance
robot = EnhancedRobotClass()

# Path to your image
image_path = 'my_picture.jpg'

# Draw the image
robot.draw_image(image_path)

# Clean up
robot.quit()