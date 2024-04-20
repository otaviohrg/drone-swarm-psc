def get_odometer_values(drone):
    '''Return odometer values if available, else return None'''
    if drone.odometer_is_disabled():
        return None, None, None
    else:
        (dist, alpha, theta) = drone.odometer_values()
        return dist, alpha, theta
