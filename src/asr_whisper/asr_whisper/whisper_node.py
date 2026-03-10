#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import torch
import torchaudio


class WhisperNode(Node):
    def __init__(self):
        super().__init__('ASR_node')
        self.publisher_ = self.create_publisher(String, '/asr/transcript', 10)
        self.q = queue.Queue(maxsize=30)
        
        # Load Whisper model
        self.get_logger().info("Loading Whisper model...")
        self.model = WhisperModel("base.en", device="cpu", compute_type="int8")
        
        # Load Silero VAD model
        self.get_logger().info("Loading VAD model...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        self.samplerate = 16000
        self.get_logger().info("ASR Node started. Speak...")
        
        # Start audio stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                blocksize=8000,
                dtype='float32',
                channels=1,
                callback=self.callback
            )
            self.stream.start()
        except Exception as e:
            self.get_logger().error(f"Failed to start audio stream: {e}")
            raise

        
        self.listen_thread = threading.Thread(target=self.listen, daemon=True)
        self.listen_thread.start()

    def callback(self, indata, frames, time, status):
        if status:
            self.get_logger().warning(str(status))
        try:
            self.q.put_nowait(indata.copy())  # Non-blocking put
        except queue.Full:
            self.get_logger().warning("Audio queue full, dropping frame")
    
    def listen(self):
        buffer = []
        is_speaking = False
        silence_frames = 0
        silence_threshold = 10  # Frames of silence before transcribing (~0.5s at 16kHz, 512 samples/frame)
        
        while rclpy.ok():
            try:
                data = self.q.get(timeout=1)
            except queue.Empty:
                # If we have buffered speech but get timeout, consider it end of speech
                if buffer and is_speaking:
                    silence_frames += 1
                    if silence_frames >= silence_threshold:
                        self._transcribe_buffer(buffer)
                        buffer = []
                        is_speaking = False
                        silence_frames = 0
                continue
            
            # Prepare audio for VAD (16-bit PCM, mono)
            audio_chunk = data.flatten()
            audio_int16 = (audio_chunk * 32768).astype(np.int16)
            
            # Run VAD on this chunk
            with torch.no_grad():
                speech_prob = self.vad_model(torch.from_numpy(audio_int16).float(), self.samplerate).item()
            
            # Speech detected if probability > threshold
            if speech_prob > 0.5:
                buffer.append(data)
                is_speaking = True
                silence_frames = 0
            else:
                # No speech detected
                if is_speaking:
                    silence_frames += 1
                    # Continue adding some trailing silence for context
                    buffer.append(data)
                    
                    # Transcribe when silence threshold is reached
                    if silence_frames >= silence_threshold:
                        self._transcribe_buffer(buffer)
                        buffer = []
                        is_speaking = False
                        silence_frames = 0
    
    def _transcribe_buffer(self, buffer):
        if not buffer:
            return
        
        audio = np.concatenate(buffer, axis=0).flatten()
        
        try:
            segments, _ = self.model.transcribe(
                audio,
                language="en",
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=700,
                    threshold=0.4,  # Lower threshold = more sensitive
                    min_speech_duration_ms=300
                ),
                beam_size=3,
                best_of=3,
                condition_on_previous_text=False
            )
            
            text = " ".join(s.text for s in segments).strip()
            
            if text:
                self.get_logger().info(f"Recognized: {text}")
                msg = String()
                msg.data = text
                self.publisher_.publish(msg)
        
        except Exception as e:
            self.get_logger().error(f"Transcription failed: {e}")

    def destroy_node(self):
        self.stream.stop()
        self.stream.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WhisperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()

if __name__ == '__main__':
    main()