from interaction import LiveSpeechToText
from task_planning import CommandGeneration
from control import Communicate

# Goal now: move robot to grab a cup and move to a different location

def main():
    # Planned to be implemented in future
    '''
    stt = LiveSpeechToText(model_path="models/vosk-model-small-en-us-0.15")                      
    text_order = stt.start()
    llm = CommandGeneration()
    command_list = llm.generate_command_list()
    command_list = [] # ["grab cup", "pour"]
    for command in command_list:
        pass
    '''

if __name__ == "__main__":
    main() 