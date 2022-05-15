import math
import time
import pygame
import numpy as np
from numpy import random
from sys import exit

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

pygame.init()

class NN():
  def __init__(self):
    self.mlp = MLPClassifier(hidden_layer_sizes=(4,4), max_iter=10000, verbose=True, tol=1e-05)

class Ball():
  def __init__(self, screen_size, radius):
    self.screen_size = screen_size
    self.radius = radius
    self.x = screen_size[0]/2 #middle
    self.y = random.randint(screen_size[1])
    self.direction = 'UP'
    self.moving = 'LEFT'
    self.speed = 5
    self.color = (255,255,255)

  def reset(self):
    self.x = math.ceil(self.screen_size[0]/2) #middle
    self.y = random.randint(self.screen_size[1])

  def move(self):
    if(self.direction == 'UP'):
      if((self.y - (self.radius/2) - self.speed) >= 0):
        self.y -= self.speed
      else:
        self.y = 0
        self.direction = 'DOWN'  
    elif(self.direction == 'DOWN'):
      if((self.y+(self.radius/2)+self.speed) <= self.screen_size[1]):
        self.y += self.speed
      else:
        self.y = math.ceil(self.screen_size[1] - (self.radius/2))
        self.direction = 'UP'

    if(self.moving == 'RIGHT'):
      if(self.x < self.screen_size[0]):
        self.x += self.speed
      else:
        self.reset()
    elif(self.moving == 'LEFT'):
      if(self.x+self.speed >= 0):
        self.x -= self.speed
      else:
        self.reset()

class Racket():
  def __init__(self, screen_size, width, height, racket = 1):
    self.screen_size = screen_size
    self.width = width
    self.height = height
    self.speed = 5
    self.color = (255,255,255)

    if racket == 1:
      self.x = 0
    elif(racket == 2):
      self.x = self.screen_size[0] - self.width
    else:
      raise Exception("racket number invalid, put 1 or 2")

    self.y = random.randint(self.screen_size[1] - height)

  def up(self):
    if(self.y > self.speed):
      self.y -= self.speed
    else:
      self.y = 0

  def down(self):
    if((self.y + self.speed + self.height) <= self.screen_size[1]):
      self.y += self.speed
    else:
      self.y = self.screen_size[1] - self.height

class Pong():
  def __init__ (self, title, width, height):
    self.title = title
    self.width = width
    self.height = height

    self.my_score = 0
    self.ai_score = 0

    self.x_train = []
    self.y_train = []

    pygame.display.set_caption(self.title)

    self.screen = pygame.display.set_mode([self.width, self.height])
    self.font = pygame.font.SysFont('Arial', 20)
    self.clock = pygame.time.Clock()

    self.racket_1 = Racket((self.width, self.height), width=20, height=100)
    self.racket_2 = Racket((self.width, self.height), width=20, height=100, racket=2)

    self.ball = Ball((self.width, self.height), 15)

    self.AI = NN()
    self.genaration = 0

  def has_colision_racket_2(self):
    if(self.ball.y in np.arange(self.racket_2.y, self.racket_2.y + self.racket_2.height)):
      self.ball.moving = 'LEFT'
    else:
      self.ai_score += 1

  def has_colision_racket_1(self):
    #print(self.ball.y, np.arange(self.racket_1.y, self.racket_1.y + self.racket_1.height))
    if(self.ball.y in np.arange(self.racket_1.y, self.racket_1.y + self.racket_1.height)):
      self.ball.moving = 'RIGHT'
    else:
      self.my_score += 1

  def make_data(self):
    if len(self.x_train) >= 5000:
      return

    #by distance
    racket_1_middle = self.racket_1.y-(self.racket_1.height/2)
    distance = self.ball.y - racket_1_middle

    self.x_train.append([distance])
    if(distance < 0): #UP
      self.y_train.append([0])
    else: #DOWN
      self.y_train.append([1])


  def fit(self):
    self.ai_score = 0
    self.my_score = 0
    self.genaration += 1

    self.x_train = np.array(self.x_train)
    self.y_train = np.array(self.y_train)

    scaler = StandardScaler()
    self.x_train = scaler.fit_transform(self.x_train)

    self.AI.mlp.fit(self.x_train, self.y_train)

    self.x_train = []
    self.y_train = []



  def play(self):
    i = 0
    while True:
      #update screen
      self.screen.fill(0)

      for event in pygame.event.get():
        if(event.type == pygame.QUIT):
          pygame.quit()
          exit()


      if(pygame.key.get_pressed()[pygame.K_UP]):
        self.racket_2.up()
      if(pygame.key.get_pressed()[pygame.K_DOWN]):
        self.racket_2.down()

      """ if(pygame.key.get_pressed()[pygame.K_w]):
        self.racket_1.up()
      if(pygame.key.get_pressed()[pygame.K_s]):
        self.racket_1.down() """

      if(self.genaration>0):
        x_in = np.array([int(self.ball.y)-int(self.racket_1.y+(self.racket_1.height/2))])

        predict = self.AI.mlp.predict([x_in])

        if(predict[0] == 0):
          self.racket_1.up()
        elif(predict[0] == 1):
          self.racket_1.down()

      #FPS
      self.clock.tick(144)
      self.screen.blit(self.font.render(str(int(self.clock.get_fps())), 1, (255,255,255)), (0, 0))

      #Divider
      pygame.draw.rect(self.screen, (255,255,255), ((self.width/2) - 2, 0, 4, self.height))

      #Points
      self.screen.blit(self.font.render(str(f'AI: {self.ai_score}'), 1, (255,255,255)), ((self.width/2) - 70, 0))
      self.screen.blit(self.font.render(str(f'Me: {self.my_score}'), 1, (255,255,255)), ((self.width/2) + 20, 0))

      #Rackets
      pygame.draw.rect(self.screen, self.racket_1.color, (self.racket_1.x, self.racket_1.y, self.racket_1.width, self.racket_1.height))
      pygame.draw.rect(self.screen, self.racket_2.color, (self.racket_2.x, self.racket_2.y, self.racket_2.width, self.racket_2.height))

      #Ball
      pygame.draw.circle(self.screen, self.ball.color, (self.ball.x, self.ball.y), self.ball.radius)

      self.ball.move()

      #pygame.draw.rect(self.screen, (255,255, 0), (self.racket_2.x, 0, self.ball.speed, self.height))
      #pygame.draw.rect(self.screen, (255,255, 0), (self.racket_1.width, 0, self.ball.speed, self.height))

      if((self.ball.moving == 'RIGHT') 
        and ((self.ball.x+self.ball.radius-self.ball.speed) in np.arange(self.racket_2.x, self.racket_2.x + self.ball.speed))):
        self.has_colision_racket_2()
        #time.sleep(3)
      
      if((self.ball.moving == 'LEFT') 
        and ((self.ball.x-self.ball.radius) in np.arange(self.racket_1.width - self.ball.speed, self.racket_1.width))):
        self.has_colision_racket_1()
        #time.sleep(3)

      i += 1
      if i == 72:
        i = 0
        self.make_data()

      if self.my_score == 5:
        self.fit()

      pygame.display.update()


if __name__ == "__main__":
  Pong("Pong", 1200, 800).play()