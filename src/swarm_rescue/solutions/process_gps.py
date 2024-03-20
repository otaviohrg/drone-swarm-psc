def get_gps_values(drone):
    '''Return GPS values if available, else use no gps strategy'''
    if drone.gps_is_disabled():
        return no_gps_strategy(drone)
    else:
        return drone.measured_gps_position()[0], drone.measured_gps_position()[1]

def no_gps_strategy(drone):
    return 0, 0