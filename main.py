from typing import BinaryIO
from visualize import draw_net
import pygame
import random
import os
import time
import neat
import pickle
pygame.font.init()

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
WIN_WIDTH = 800
WIN_HEIGHT = 700
FLOOR = 0
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Car Control")
gen = 0

bg = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","background.jpg")).convert_alpha())
car_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","car.png")).convert_alpha())
obs_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","barrier.png")).convert_alpha())
road_temp_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","road.jpg")).convert_alpha())
road_img = pygame.transform.rotate(road_temp_img, 90)

class Car:
    IMG = car_img
    ANIMATION_TIME = 5
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.height = self.IMG.get_height()
        self.img = self.IMG

    def move_left(self):
        if self.x >= 240:
            self.x -= 185
            return True

    def move_right(self):
        if self.x <= 240:
            self.x += 185
            return True

    def move(self):
        if self.move_right():
            return
        else: self.move_left()

    def draw(self, win):
        win.blit(self.IMG, (self.x, self.y))

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Obs:
    IMG = obs_img
    VEL = 10
    HEIGHT = obs_img.get_height()

    def __init__(self, flag):
        if flag == 0:
            self.x = 55
        elif flag == 1:
            self.x = 240
        else: self.x = 425
        self.y = 0
        self.passed = False

    def move(self):
        self.y += self.VEL
    
    def draw(self, win):
        win.blit(self.IMG, (self.x, self.y))

    def collide(self, car, win):
        """
        returns if a point is colliding with the obs
        :param car: Car object
        :return: Bool
        """
        car_mask = car.get_mask()
        obs_mask = pygame.mask.from_surface(obs_img)
        offset = (self.x - car.x, self.y - round(car.y))

        check = car_mask.overlap(obs_mask, offset)
        if check:
            return True
        return False
 

class Base:
    VEL = 10
    HEIGHT = road_img.get_height()
    IMG = road_img

    def __init__(self, x):
        self.x = x
        self.y1 = 0
        self.y2 = self.HEIGHT
    
    def move(self):
        self.y1 += self.VEL
        self.y2 += self.VEL
        if self.y1 > self.HEIGHT:
            self.y1 = self.y2 - self.HEIGHT

        if self.y2 > self.HEIGHT:
            self.y2 = self.y1 - self.HEIGHT
    
    def draw(self, win):
        win.blit(self.IMG, (self.x, self.y1))
        win.blit(self.IMG, (self.x, self.y2))

class BG:
    def __init__(self):
        self.IMG = bg
    def draw(self, win):
        win.blit(self.IMG, (0, 0))
        
def drawWindow(win, cars, road, obses, score, gen):
    road.draw(win)
    for car in cars:
        car.draw(win)
    for obs in obses:
        obs.draw(win)
    score_label = STAT_FONT.render("Score: " + str(score),1,(0,255,0))
    gen_label = STAT_FONT.render("Gens: " + str(gen-1),1,(255,255,255))
    alive_label = STAT_FONT.render("Alive: " + str(len(cars)),1,(255,255,255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))
    win.blit(gen_label, (10, 50))
    win.blit(alive_label, (10, 10))
    pygame.display.update()

def eval(genomes, config):
    global gen
    gen += 1
    nets = []
    cars = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car(240, 500))
        ge.append(genome)

    road = Base(FLOOR)
    bg = BG()
    obses = [Obs(random.randint(0, 2))]
    score = 0

    clock = pygame.time.Clock()
    run = True
    state = True
    while run and len(cars) > 0:
        #draw_net(config, genome)
        clock.tick(30)
        bg.draw(WIN)

        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    car.move_left()
                if event.key == pygame.K_RIGHT:
                    car.move_right()
                if event.key == pygame.K_p:
                    state = not state
        if state:
            obs_ind = 0
            if len(cars) > 0:
                if len(obses) > 1 and cars[0].y < obses[0].y - cars[0].height:
                    obs_ind = 1

            for x, car in enumerate(cars):
                ge[x].fitness += 0.1
                if len(obses) > 1:
                    output = nets[cars.index(car)].activate((car.x, 
                    abs(car.x - obses[obs_ind].x), abs(car.y - obses[obs_ind].y), 
                    abs(car.x - obses[1].x), abs(car.y - obses[1].y)))
                    if output[0] > 0.5 :
                        car.move_right()
                    elif output[1] > 0.5:
                        car.move_left()

            road.move()

            add_obs = False
            remover_obs = []
            for obs in obses:
                obs.move()
                for car in cars:
                    if obs.collide(car, WIN):
                        ge[cars.index(car)].fitness -= 1
                        nets.pop(cars.index(car))
                        ge.pop(cars.index(car))
                        cars.pop(cars.index(car))
                if obs.y > 700:
                    remover_obs.append(obs)

                if not obs.passed and obs.y > car.y - obs.HEIGHT:
                        obs.passed = True
                        add_obs = True

            if add_obs:
                score += 1
                for genome in ge:
                    genome.fitness += 1
                obses.append(Obs(random.randint(0,2)))

            for r in remover_obs:
                obses.remove(r)

            for car in cars:
                if car.x > 600 or car.x < 0:
                    nets.pop(cars.index(car))
                    ge.pop(cars.index(car))
                    cars.pop(cars.index(car))
            
            drawWindow(WIN, cars, road, obses, score, gen)
            if score > 100:
                pickle.dump(nets[0],open("best.pickle", "wb"))
                break

def rerun(config_file, pre_genomes = "winner.pkl"):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    with open(pre_genomes, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    # Call game with only the loaded genome
    eval(genomes, config)


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval, 50)
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)