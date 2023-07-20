import pygame, time, random
from pygame import mixer
pygame.mixer.init()
cardsounds = ["Cards\Resources\Card1.mp3", "Cards\Resources\Card2.mp3", "Cards\Resources\Card3.mp3", "Cards\Resources\Card4.mp3", "Cards\Resources\Card5.mp3", "Cards\Resources\Card6.mp3","Cards\Resources\CardFull.mp3","Cards\Resources\Cardsmooth.mp3"]

for i in range(len(cardsounds)):
    pygame.mixer.music.load(cardsounds[i])
pygame.init()
width = 1200
height = 800
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
card_x = 150
card_y = 360 - 72
def lay_cards():

    for i in range(10):
        i = pygame.draw.rect(screen, (170, 170, 170), (card_x, card_y, 140, 140), 2, 5)
        while i.x < 600:
            i.x += 25
            i.y += random.randint(-5,-2)
            screen.fill((255, 255, 255))
            i = pygame.draw.rect(screen, (170, 170, 170), (i.x, i.y, 140, 140), 2, 5)
            pygame.display.update()
        while i.x < 1200:
            i.x += 25
            i.y += random.randint(2,5)
            screen.fill((255, 255, 255))
            i = pygame.draw.rect(screen, (170, 170, 170), (i.x, i.y, 140, 140), 2, 5)
            pygame.display.update()
        a = pygame.draw.rect(screen, (170, 170, 170), (i.x, i.y, 140, 140), 2,5)
        pygame.display.update()
    screen.fill((255, 255, 255))
    spread_cards()
def waitload():
    pygame.mixer.music.play(random.randint(1,6))
    pygame.display.update()
    time.sleep(0.1)

def spread_cards():
    a = pygame.draw.rect(screen, (170, 170, 170), (530,330, 140, 140), 2, 5)
    waitload()
    b = pygame.draw.rect(screen, (170, 170, 170), (530+140,330, 140, 140), 2, 5)
    waitload()
    c = pygame.draw.rect(screen, (170, 170, 170), (530+140,330-140, 140, 140), 2, 5)
    waitload()
    d = pygame.draw.rect(screen, (170, 170, 170), (530,330-140, 140, 140), 2, 5)
    waitload()
    e = pygame.draw.rect(screen, (170, 170, 170), (530-140,330-140, 140, 140), 2, 5)
    waitload()
    f = pygame.draw.rect(screen, (170, 170, 170), (530-140,330, 140, 140), 2, 5)   
    waitload()
    g = pygame.draw.rect(screen, (170, 170, 170), (530-140,330+140, 140, 140), 2, 5)   
    waitload()
    h = pygame.draw.rect(screen, (170, 170, 170), (530,330+140, 140, 140), 2, 5)   
    waitload()
    j = pygame.draw.rect(screen, (170, 170, 170), (530+140,330+140, 140, 140), 2, 5) 
    waitload()
    k = pygame.draw.rect(screen, (170, 170, 170), (530+280,330+140, 140, 140), 2, 5) 
    waitload()
    l = pygame.draw.rect(screen, (170, 170, 170), (530+280,330, 140, 140), 2, 5) 
    waitload()
    m = pygame.draw.rect(screen, (170, 170, 170), (530+280,330-140, 140, 140), 2, 5)
    waitload()
    n = pygame.draw.rect(screen, (170, 170, 170), (530+280,330-280, 140, 140), 2, 5)
    waitload()
    o = pygame.draw.rect(screen, (170, 170, 170), (530+140,330-280, 140, 140), 2, 5) 
    waitload()
    p = pygame.draw.rect(screen, (170, 170, 170), (530+140,330-280, 140, 140), 2, 5) 
    waitload()
    q = pygame.draw.rect(screen, (170, 170, 170), (530,330-280, 140, 140), 2, 5) 
    waitload()
    r = pygame.draw.rect(screen, (170, 170, 170), (530-140,330-280, 140, 140), 2, 5)
    waitload()
    s = pygame.draw.rect(screen, (170, 170, 170), (530-280,330-280, 140, 140), 2, 5)
    waitload()
    t = pygame.draw.rect(screen, (170, 170, 170), (530-280,330-140, 140, 140), 2, 5) 
    waitload()
    u = pygame.draw.rect(screen, (170, 170, 170), (530-280,330, 140, 140), 2, 5) 
    waitload()
    v=pygame.draw.rect(screen, (170, 170, 170), (530-280,330+140, 140, 140), 2, 5)
    waitload()
    w=pygame.draw.rect(screen, (170, 170, 170), (530-280,330+280, 140, 140), 2, 5) 
    waitload()
    x=pygame.draw.rect(screen, (170, 170, 170), (530-140,330+280, 140, 140), 2, 5)
    waitload()
    y=pygame.draw.rect(screen, (170, 170, 170), (530,330+280, 140, 140), 2, 5) 
    waitload()
    z=pygame.draw.rect(screen, (170, 170, 170), (530+140,330+280, 140, 140), 2, 5) 
    waitload()
    zy=pygame.draw.rect(screen, (170, 170, 170), (530+280,330+280, 140, 140), 2, 5)
    waitload()
while True:
    clock.tick(60)
    screen.fill((255, 255, 255))
    circle = pygame.draw.circle(screen, (255, 0, 0), (600, 50),35)
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if circle.collidepoint(event.pos):
                pygame.mixer.music.play(4,0,8)
                lay_cards()
                pygame.mixer.music.stop()
                time.sleep(1)
        pygame.display.update()

