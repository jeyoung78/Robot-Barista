# Goal now: move robot to grab a cup and move to a different location
# Monday: Implement robot code and Communicate class so that robot can be controlled from python script

from control import Communicate
from task_planning import PolicyGeneration


def main():
    query = "Q: To make coffee what action should a robot take first?\nA:"
    options = [
        " locate espresso machine.",
        " place cup below espresso machine.",
        " press button brew coffee from espresso machine.",
        " put coffee into espresso machine."
    ]
    policy_generation = PolicyGeneration()
    scores = policy_generation.local_llm_scoring(query, options, option_start="\n", verbose=False)
    print(scores)

def initial_alingment():
    pass

def grab_cup(): 
    pass
    
if __name__ == "__main__":
    main() 