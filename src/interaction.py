import numpy as np
import sounddevice as sd
import keyboard
from vosk import Model, KaldiRecognizer
import json
from queue import Queue

class LiveSpeechToText:
    def __init__(self, model_path="models/vosk-model-en-us-0.22", sample_rate=16000):
        self.model = Model(model_path)
        self.sample_rate = sample_rate
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.is_recording = False
        self.audio_buffer = []
        self.stream = None
        self.result_queue = Queue()
        self.final_results = []

    def _audio_callback(self, indata, frames, time, status):
        self.audio_buffer.append(indata.copy())

    def _toggle_recording(self):
        if not self.is_recording:
            # Start recording
            print("Listening...")
            self.audio_buffer = []
            self.is_recording = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16,
                callback=self._audio_callback
            )
            self.stream.start()
        else:
            # Stop recording and process
            print("Processing...")
            self.stream.stop()
            self.stream.close()
            self.is_recording = False

            audio_data = np.concatenate(self.audio_buffer)
            audio_bytes = audio_data.tobytes()
            
            self.recognizer.Reset()
            if self.recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self.recognizer.Result())
                self.result_queue.put(result['text'])
            else:
                result = json.loads(self.recognizer.PartialResult())
                self.result_queue.put(result['partial'])

            # print(self.result_queue)

    def start(self):
        """
        Start listening for spacebar presses
        Returns: List of transcribed texts in order
        """
        keyboard.add_hotkey('space', self._toggle_recording)
        print("Press SPACE to start/stop recording. Press ESC to exit.")
        keyboard.wait('esc')
        
        # Collect all results from the queue
        while not self.result_queue.empty():
            self.final_results.append(self.result_queue.get())
            # print(self.final_results)
            
        return ' '.join(self.final_results)

if __name__ == "__main__":
    stt = LiveSpeechToText(model_path="models/vosk-model-en-us-0.22")
    results = stt.start()
    print("All transcriptions:", results) 