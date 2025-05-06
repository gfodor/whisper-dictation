import argparse
import time
import threading
import pyaudio
import numpy as np
from whisper import load_model
import platform
import math

def play_tone(frequency, duration=0.067, volume=0.3):
    """Play a tone with the given frequency, duration, and volume."""
    try:
        p = pyaudio.PyAudio()
        sample_rate = 44100  # samples per second
        
        # Generate samples
        samples = (np.sin(2 * np.pi * np.arange(sample_rate * duration) * frequency / sample_rate)).astype(np.float32)
        samples = samples * volume
        
        # Apply envelope to avoid clicks
        envelope = np.ones_like(samples)
        ramp_samples = int(0.015 * sample_rate)  # 15ms ramp
        if ramp_samples * 2 < len(samples):
            envelope[:ramp_samples] = np.linspace(0, 1, ramp_samples)
            envelope[-ramp_samples:] = np.linspace(1, 0, ramp_samples)
        else:
            mid_point = len(samples) // 2
            envelope[:mid_point] = np.linspace(0, 1, mid_point)
            envelope[mid_point:] = np.linspace(1, 0, len(samples) - mid_point)
        
        samples = samples * envelope
        
        # Complete waveform cycle to avoid clicks
        sample_length = len(samples)
        cycles = frequency * duration
        if not math.isclose(cycles, round(cycles), abs_tol=0.1):
            last_sample_idx = int(round(cycles) * sample_rate / frequency)
            if last_sample_idx < sample_length:
                fade_len = sample_length - last_sample_idx
                fade_envelope = np.linspace(1, 0, fade_len)
                samples[last_sample_idx:] = samples[last_sample_idx:] * fade_envelope
        
        # Convert to int16
        samples = (samples * 32767).astype(np.int16)
        
        # Open and play stream
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        output=True)
        stream.write(samples.tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()
    except Exception as e:
        print(f"Error playing tone: {e}")

class SpeechTranscriber:
    def __init__(self, model):
        self.model = model

    def transcribe(self, audio_data, language=None):
        result = self.model.transcribe(audio_data, language=language)
        transcription_text = result["text"].lstrip() # Remove leading space if present
        print(f"Transcription: {transcription_text}") # Log transcription
        return transcription_text

from pynput import keyboard # Add keyboard import here
import time # Ensure time is imported if not already globally available for sleep

class Recorder:
    def __init__(self, transcriber):
        self.recording = False
        self.transcriber = transcriber
        self.pykeyboard = keyboard.Controller() # Add keyboard controller instance

    def start(self, language=None):
        thread = threading.Thread(target=self._record_impl, args=(language,))
        thread.start()

    def stop(self):
        self.recording = False

    def _record_impl(self, language):
        self.recording = True
        frames_per_buffer = 1024
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        frames_per_buffer=frames_per_buffer,
                        input=True)
        frames = []

        while self.recording:
            data = stream.read(frames_per_buffer)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data_fp32 = audio_data.astype(np.float32) / 32768.0
        # Get transcription text
        transcription = self.transcriber.transcribe(audio_data_fp32, language)
        # Type the transcription
        if transcription:
            try:
                for element in transcription:
                    self.pykeyboard.type(element)
                    time.sleep(0.0025) # Keep the small delay
                print("Typing complete.")
            except Exception as e:
                print(f"Error typing transcription: {e}")
        else:
            print("No transcription text to type.")

class RecordingManager:
    def __init__(self, recorder, language, max_time):
        self.recorder = recorder
        self.language = language
        self.max_time = max_time
        self.recording = False
        self.timer = None

    def start(self):
        if not self.recording:
            print("Listening...")
            self.recording = True
            self.recorder.start(self.language)
            if self.max_time is not None:
                self.timer = threading.Timer(self.max_time, self.stop)
                self.timer.start()

    def stop(self):
        if self.recording:
            if self.timer is not None:
                self.timer.cancel()
            print("Transcribing...")
            self.recording = False
            self.recorder.stop()
            print("Done.\n")

    def toggle(self):
        if self.recording:
            self.stop()
        else:
            self.start()

class GlobalKeyListener:
    def __init__(self, recording_manager, key_combination):
        self.recording_manager = recording_manager
        self.key1, self.key2 = self.parse_key_combination(key_combination)
        self.key1_pressed = False
        self.key2_pressed = False

    def parse_key_combination(self, key_combination):
        key1_name, key2_name = key_combination.split('+')
        key1 = getattr(keyboard.Key, key1_name, keyboard.KeyCode(char=key1_name))
        key2 = getattr(keyboard.Key, key2_name, keyboard.KeyCode(char=key2_name))
        return key1, key2

    def on_key_press(self, key):
        if key == self.key1:
            self.key1_pressed = True
        elif key == self.key2:
            self.key2_pressed = True
        if self.key1_pressed and self.key2_pressed:
            self.recording_manager.toggle()

    def on_key_release(self, key):
        if key == self.key1:
            self.key1_pressed = False
        elif key == self.key2:
            self.key2_pressed = False

class DoubleCommandKeyListener:
    def __init__(self, recording_manager):
        self.recording_manager = recording_manager
        self.key = keyboard.Key.cmd_r
        self.last_press_time = 0

    def on_key_press(self, key):
        if key == self.key:
            current_time = time.time()
            if not self.recording_manager.recording and current_time - self.last_press_time < 0.5:
                self.recording_manager.start()
            elif self.recording_manager.recording:
                self.recording_manager.stop()
            self.last_press_time = current_time

    def on_key_release(self, key):
        pass

class PushToTalkListener:
    def __init__(self, recording_manager):
        self.recording_manager = recording_manager
        self.key = keyboard.Key.cmd_l
        self.active = False
        self.last_press_time = 0

    def on_key_press(self, key):
        if key == self.key:
            current_time = time.time()
            if not self.active and current_time - self.last_press_time < 0.5:
                self.active = True
                threading.Thread(target=play_tone, args=(300,)).start()
                self.recording_manager.start()
            self.last_press_time = current_time

    def on_key_release(self, key):
        if key == self.key and self.active:
            self.active = False
            threading.Thread(target=play_tone, args=(600,)).start()
            self.recording_manager.stop()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Dictation app using the OpenAI whisper ASR model. By default the keyboard shortcut cmd+option '
                    'starts and stops dictation')
    parser.add_argument('-m', '--model_name', type=str,
                        choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large'],
                        default='base',
                        help='Specify the whisper ASR model to use.')
    parser.add_argument('-k', '--key_combination', type=str, default='cmd_l+alt' if platform.system() == 'Darwin' else 'ctrl+alt',
                        help='Key combination to toggle recording.')
    parser.add_argument('--k_double_cmd', action='store_true',
                        help='Use double Right Command key press to start recording, single press to stop.')
    parser.add_argument('--ptt', action='store_true',
                        help='Use double tap of Left Command key to activate push-to-talk mode.')
    parser.add_argument('-l', '--language', type=str, default=None,
                        help='Specify the two-letter language code (e.g., "en" for English).')
    parser.add_argument('-t', '--max_time', type=float, default=30,
                        help='Maximum recording time in seconds.')
    parser.add_argument('--audio_file', type=str, default=None,
                       help='Path to an audio file (.wav or .webm) to transcribe directly. If provided, other options are ignored.')
    args = parser.parse_args()

    if args.language is not None:
        args.language = args.language.split(',')
    if args.model_name.endswith('.en') and args.language is not None and any(lang != 'en' for lang in args.language):
        raise ValueError('If using a .en model, language must be English.')
    return args

if __name__ == "__main__":
    args = parse_args()

    # Check if transcribing a specific audio file (WAV or WEBM)
    if args.audio_file:
        print(f"Loading model ({args.model_name})...")
        model = load_model(args.model_name)
        print(f"{args.model_name} model loaded.")
        language = args.language[0] if args.language else None
        print(f"Transcribing file: {args.audio_file}...")
        try:
            result = model.transcribe(args.audio_file, language=language)
            transcription = result["text"].lstrip()
            print("\n--- Transcription ---")
            print(transcription)
            print("--- End Transcription ---\n")
        except FileNotFoundError:
            print(f"Error: Audio file not found at {args.audio_file}")
        except Exception as e:
            print(f"Error during transcription: {e}")
        exit() # Exit after transcribing the file

    # --- Original execution flow for live dictation ---

    # Play startup tone
    threading.Thread(target=play_tone, args=(800, 0.3, 0.5)).start()

    print("Loading model...")
    model = load_model(args.model_name)
    print(f"{args.model_name} model loaded")
    threading.Thread(target=play_tone, args=(500, 0.2, 0.4)).start()

    transcriber = SpeechTranscriber(model)
    recorder = Recorder(transcriber)
    language = args.language[0] if args.language else None
    recording_manager = RecordingManager(recorder, language, args.max_time)

    if args.ptt:
        key_listener = PushToTalkListener(recording_manager)
    elif args.k_double_cmd:
        key_listener = DoubleCommandKeyListener(recording_manager)
    else:
        key_listener = GlobalKeyListener(recording_manager, args.key_combination)

    listener = keyboard.Listener(on_press=key_listener.on_key_press, on_release=key_listener.on_key_release)
    listener.start()

    print("Running... Press Ctrl+C to exit.")
    try:
        listener.join()
    except KeyboardInterrupt:
        print("\nExiting...")
        listener.stop()
        if recording_manager.recording:
            recording_manager.stop()
