from enum import Enum

class Activity(Enum):
    """
    All the states of the drone as a state machine
    """
    SEARCHING_WOUNDED = 1
    GRASPING_WOUNDED = 2
    SEARCHING_RESCUE_CENTER = 3
    DROPPING_AT_RESCUE_CENTER = 4

def update_state(activity, found_wounded, found_rescue_center, grasped_entities, voteResult):

    if activity is Activity.SEARCHING_WOUNDED and found_wounded and voteResult:
        return Activity.GRASPING_WOUNDED
    
    elif activity is Activity.GRASPING_WOUNDED and (not voteResult):
        return Activity.SEARCHING_WOUNDED

    elif activity is Activity.GRASPING_WOUNDED and grasped_entities:
        return Activity.SEARCHING_RESCUE_CENTER

    elif activity is Activity.GRASPING_WOUNDED and not found_wounded:
        return Activity.SEARCHING_WOUNDED

    elif activity is Activity.SEARCHING_RESCUE_CENTER and found_rescue_center:
        return Activity.DROPPING_AT_RESCUE_CENTER

    elif activity is Activity.DROPPING_AT_RESCUE_CENTER and not grasped_entities:
        return Activity.SEARCHING_WOUNDED

    elif activity is Activity.DROPPING_AT_RESCUE_CENTER and not found_rescue_center:
        return Activity.SEARCHING_RESCUE_CENTER
    
    else:
        return Activity.GRASPING_WOUNDED
