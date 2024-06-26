"""
This file was generated by the tool 'image_to_map.py' in the directory tools.
This tool permits to create this kind of file by providing it an image of the map we want to create.
"""

from spg_overlay.entities.normal_wall import NormalWall, NormalBox


# Dimension of the map : (800, 500)
# Dimension factor : 1.0
def add_boxes(playground):
    pass


def add_walls(playground):
    # vertical wall 0
    wall = NormalWall(pos_start=(-395, 248),
                      pos_end=(-395, -247))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 1
    wall = NormalWall(pos_start=(-196, 247),
                      pos_end=(-196, -94))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 2
    wall = NormalWall(pos_start=(-193, 248),
                      pos_end=(-193, -93))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 3
    wall = NormalWall(pos_start=(-394, 245),
                      pos_end=(397, 245))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 4
    wall = NormalWall(pos_start=(61, 85),
                      pos_end=(61, -85))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 5
    wall = NormalWall(pos_start=(59, -81),
                      pos_end=(397, -81))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 6
    wall = NormalWall(pos_start=(393, 247),
                      pos_end=(393, -247))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 7
    wall = NormalWall(pos_start=(64, 84),
                      pos_end=(64, -82))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 8
    wall = NormalWall(pos_start=(61, -78),
                      pos_end=(396, -78))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 9
    wall = NormalWall(pos_start=(-397, -243),
                      pos_end=(396, -243))
    playground.add(wall, wall.wall_coordinates)

