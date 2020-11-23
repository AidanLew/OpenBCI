import pygame


class left_right():

    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((640, 240))
        self.x = 320
        self.update()

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
        self.display.fill('gray')  # Background
        pygame.draw.line(self.display, 'black', (80, 120), (560, 120), 5)  # Line Accross
        pygame.draw.line(self.display, 'black', (320, 105), (320, 135), 5)  # Line Up 0
        pygame.draw.line(self.display, 'black', (80, 100), (80, 140), 5)  # Line Up -100
        pygame.draw.line(self.display, 'black', (560, 100), (560, 140), 5)  # Line Up 100
        pygame.draw.line(self.display, 'black', (200, 110), (200, 130), 4)  # Line Up -50
        pygame.draw.line(self.display, 'black', (440, 110), (440, 130), 4)  # Line Up 50
        pygame.draw.circle(self.display, 'red', (50, 120), 20)  # Red Circle
        pygame.draw.rect(self.display, 'blue', (570, 100, 40, 40))  # Blue Square
        pygame.draw.circle(self.display, 'white', (self.x, 120), 8)  # White Circle
        pygame.display.flip()

    def move(self, x):
        self.x += x
        if self.x < 80:
            self.x = 80
        elif self.x > 560:
            self.x = 560
