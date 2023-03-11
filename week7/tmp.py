import math

from tensorboardX import SummaryWriter

writer = SummaryWriter(logdir='scalar/sin&cos')
for step in range(-360 ,360):
    angle_rad = step*math.pi / 360
    writer.add_scalar('sin', math.sin(angle_rad), step)
    writer.add_scalar('cos', math.cos(angle_rad), step)
writer.close()