#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading


class WhisperNode(Node):
    def __init__(self):
        super().__init__('ASR_node')
        self.publisher_ = self.create_publisher(String, '/asr/transcript', 10)
        self.q = queue.Queue(maxsize=30)
        
        # Load Whisper model
        self.get_logger().info("Loading Whisper model...")
        self.model = WhisperModel("base.en", device="cpu", compute_type="int8")
        
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
        chunk_count = 0
        process_every = 8  # Process every 4 seconds (8 * 0.5s)
        
        while rclpy.ok():
            try:
                data = self.q.get(timeout=1)
            except queue.Empty:
                continue
            buffer.append(data)
            chunk_count += 1
            
            # Process when we have enough audio
            if chunk_count >= process_every:
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
                
                # Clear buffer
                buffer = []
                chunk_count = 0

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