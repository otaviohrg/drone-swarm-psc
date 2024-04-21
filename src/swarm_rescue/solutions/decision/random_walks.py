import random
import math
    
def ballistic(x, y, step_distance):
    # Generate a random direction
    theta = random.uniform(0, 2*math.pi)
    # Calculate new position
    dx = step_distance * math.cos(theta)
    dy = step_distance * math.sin(theta)
    x += dx
    y += dy
    return x, y

def levy_flight(x, y, max_jump_length):
    # Generate a random jump length following a Levy distribution
    jump_length = random.expovariate(1.5) * max_jump_length
    # Generate a random direction
    theta = random.uniform(0, 2*math.pi)
    # Calculate new position
    dx = jump_length * math.cos(theta)
    dy = jump_length * math.sin(theta)
    x += dx
    y += dy
    return x, y
