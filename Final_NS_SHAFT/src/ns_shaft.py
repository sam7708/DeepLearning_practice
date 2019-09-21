"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
from itertools import cycle
from numpy.random import randint
from pygame import Rect, init, time, display
from pygame.event import pump
from pygame.image import load
from pygame.surfarray import array3d, pixels_alpha
from pygame.transform import rotate
import numpy as np


class NS_SHAFT(object):
    init()
    fps_clock = time.Clock()
    screen_width = 288
    screen_height = 512
    screen = display.set_mode((screen_width, screen_height))
    display.set_caption('Deep Q-Network NS-SH')
    base_image = load('assets/sprites/base.png').convert_alpha()
    background_image = load('assets/sprites/background-black.png').convert()

    floor_images = [load('assets/sprites/floors.PNG').convert_alpha()]
    man_images = [load('assets/sprites/man.PNG').convert_alpha()]
    # number_images = [load('assets/sprites/{}.png'.format(i)).convert_alpha() for i in range(10)]

    man_hitmask = [pixels_alpha(image).astype(bool) for image in man_images]
    floor_hitmask = [pixels_alpha(image).astype(bool) for image in floor_images]

    fps = 30
    #pipe_gap_size = 100
    floor_velocity_y = 4

    # parameters for man
    min_velocity_y = -4
    max_velocity_y = 10
    downward_speed = 1
    upward_speed = -9

    man_index_generator = cycle([0, 1, 2, 1])
    state = 0 # 0 falling, 1 stand on floor 
    state2_counter = 0

    def __init__(self):

        self.iter = self.man_index = self.score = 0

        self.man_width = self.man_images[0].get_width()
        self.man_height = self.man_images[0].get_height()
        self.floor_width = self.floor_images[0].get_width()
        self.floor_height = self.floor_images[0].get_height()

        self.man_x = int((self.screen_width- self.man_width)/2)
        #self.man_y = int((self.screen_height - self.man_height) / 2)
        self.man_y = int(self.screen_height*0.25)-50
        #self.base_x = 0
        #self.base_y = self.screen_height * 0.79
        #self.base_shift = self.base_image.get_width() - self.background_image.get_width()

        floors = [{'x':self.man_x,'y':self.man_y+self.man_height+3},{'x':30,'y':250},{'x':100,'y':400}]
        #floors[0]["x_upper"] = pipes[0]["x_lower"] = self.screen_width
        #floors[1]["x_upper"] = pipes[1]["x_lower"] = self.screen_width * 1.5
        self.floors = floors

        self.current_velocity_y = 0
        self.is_flapped = False

    def generate_floor(self):
        y = self.screen_height
        f = [randint(10,210),randint(0,75),randint(150,220)]
        x = f[randint(0,2)]
        #gap_y = randint(2, 10) * 10 + int(self.base_y / 5)
        return {'x' : x,'y' : y}

    def is_collided(self):
        man_bbox = Rect(self.man_x, self.man_y, self.man_width, self.man_height)
        collided_floor = {'x': 0,'y':0}
        is_collided = False
        for floor in self.floors:
            floor_box = Rect(floor["x"], floor["y"], self.floor_width, self.floor_height)
            if Rect.colliderect(man_bbox, floor_box):
            	collided_floor['x'] = floor['x']
            	collided_floor['y'] = floor['y']
            	is_collided = True

        return is_collided, collided_floor

    def next_frame(self, action):
        pump()
        reward = 0.01
        terminal = False
        # Check input action
        self.current_velocity_x = action

        if self.is_flapped:
            self.is_flapped = False
        if self.state == 1:
            self.current_velocity_y = self.floor_velocity_y
        elif self.state == 2:
        	self.current_velocity_y = -4
        else:
            self.current_velocity_y = max(min(-1,self.current_velocity_y-1),self.min_velocity_y)
        #print(self.state,self.current_velocity_y)
        self.man_y -= self.current_velocity_y
        self.man_x = min(max(0,self.man_x+self.current_velocity_x),self.screen_width-self.man_width)
        if(self.man_x == 0 or self.man_x == self.screen_width-self.man_width):
            reward -=1
        # Update pipes' position
        for floor in self.floors:
            #print(floor["y"],self.floor_velocity_y)
            floor["y"] -= self.floor_velocity_y
        # Update pipes
        if self.floors[0]["y"] < 0:
            self.floors.append(self.generate_floor())
            del self.floors[0]
        
        is_collided, floor = self.is_collided()
        if (self.state2_counter==0):
	        if self.man_y >=self.screen_height:
	        	self.state = 3
	        	terminal = True
	        	reward = -1
	        	self.__init__()
	        elif self.man_y <=0:
	        	self.state = 2
	        	self.state2_counter = 10
	        	reward = -0.2
	        elif is_collided:
	            self.man_y = floor['y']-28
	            self.state = 1
	            reward = 0.4
	        else:
	            self.state = 0
	            #reward = -1
        else:
            self.state2_counter = max(0,self.state2_counter-1)

        # Draw everything
        self.screen.blit(self.background_image, (0, 0))
        #self.screen.blit(self.base_image, (self.base_x, self.base_y))
        self.screen.blit(self.man_images[0], (self.man_x, self.man_y))
        for floor in self.floors:
            self.screen.blit(self.floor_images[0], (floor["x"], floor["y"]))
            #self.screen.blit(self.pipe_images[1], (pipe["x_lower"], pipe["y_lower"]))
        image = array3d(display.get_surface())
        display.update()
        self.fps_clock.tick(self.fps)
        return image, reward, terminal

#game_state = NS_SHIFT()
#for i in range(770):
#	image, reward, terminal = game_state.next_frame(0)
