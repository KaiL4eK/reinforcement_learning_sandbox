import sys
import pygame
from pygame.locals import *
import pymunk
from pymunk.pygame_util import DrawOptions
import random

def add_ball(space):
    """Add a ball to the given space at a random position"""
    mass = 1
    radius = 14
    inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
    body = pymunk.Body(mass, inertia)
    x = random.randint(120,380)
    body.position = x, 550
    shape = pymunk.Circle(body, radius, (0,0))
    space.add(body, shape)
    return shape

def add_L(space):
    """Add a inverted L shape with two joints"""
    plate_body = pymunk.Body(10, 10000)
    plate_body.position = (300,300)
    plate = pymunk.Segment(plate_body, (-150, 0), (150.0, 0.0), 3.0)

    beam_body = pymunk.Body(10, 10000)
    beam_body.position = (200,250)
    beam = pymunk.Segment(beam_body, (0, -50), (0, 50), 1.0)
    # l1 = pymunk.Segment(body, (-150, 0), (255.0, 0.0), 3.0)
    # l2 = pymunk.Segment(body, (-150.0, 0), (-150.0, 50.0), 3.0)

    # beam_center_body = pymunk.Body(body_type = pymunk.Body.STATIC)
    # beam_center_body.position = (200,300)
    # beam_center_joint = pymunk.PinJoint(plate_body, beam_body, (0,0), (0,0))

    rotation_center_body = pymunk.Body(body_type = pymunk.Body.STATIC)
    rotation_center_body.position = (300,300)
    rotation_center_joint = pymunk.PinJoint(plate_body, rotation_center_body, (0,0), (0,0))

    # rotation_limit_body = pymunk.Body(body_type = pymunk.Body.STATIC)
    # rotation_limit_body.position = (200,300)
    # joint_limit = 25
    # rotation_limit_joint = pymunk.SlideJoint(body, rotation_limit_body, (-100,0), (0,0), 0, joint_limit)

    space.add(plate, plate_body, beam, beam_body, rotation_center_body)
    return plate_body

def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Joints. Just wait and the L will tip over")
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = (0.0, -900.0)

    lines = add_L(space)
    balls = []
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    ticks_to_next_ball = 10
    direction = 1
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)

        ticks_to_next_ball -= 1
        if ticks_to_next_ball <= 0:
            ticks_to_next_ball = 25
            # lines[2].apply_force_at_local_point((0, 1000), point=(100 * direction, 0))
            # direction *= -1
            # ball_shape = add_ball(space)
            # balls.append(ball_shape)

        screen.fill((255,255,255))

        balls_to_remove = []
        for ball in balls:
            if ball.body.position.y < 150:
                balls_to_remove.append(ball)

        for ball in balls_to_remove:
            space.remove(ball, ball.body)
            balls.remove(ball)

        space.debug_draw(draw_options)

        space.step(1/50.0)

        pygame.display.flip()
        clock.tick(50)

if __name__ == '__main__':
    main()
    