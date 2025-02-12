# Goal now: move robot to grab a cup and move to a different location
# Monday: Implement robot code and Communicate class so that robot can be controlled from python script

from control import Communicate
from llm_highlevel import LLMScoring

def main():
    # query = "Human: I want you to bring me the rice chips from the drawer? Robot: To do this, the first thing I would do is to\n"
    query = "Human: I spilled my coke, can you bring me something to clean it up? Robot: To do this, the first thing I would do is to\n"
    policy_generation = LLMScoring()
    scores = policy_generation.local_llm_scoring(query, options=policy_generation.options, option_start="\n", verbose=False)
    print(scores)
    
if __name__ == "__main__":
    main() 