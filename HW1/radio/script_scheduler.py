import schedule
import time
import subprocess
import sys
import logging
from datetime import datetime
import os
from threading import Thread
import queue

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('script_runner.log'),
        logging.StreamHandler()
    ]
)

class ScriptRunner:
    def __init__(self, script_path):
        self.script_path = script_path
        self.process = None
        self.should_run = True
        self.log_queue = queue.Queue()

    def handle_output(self, pipe, is_error=False):
        """Handle output from the subprocess"""
        for line in iter(pipe.readline, b''):
            line = line.decode('utf-8').strip()
            if line:
                if is_error:
                    logging.error(f"Script error: {line}")
                else:
                    logging.info(f"Script output: {line}")
                self.log_queue.put(line)

    def start_script(self):
        """Start the target script as a subprocess"""
        try:
            logging.info(f"Starting script: {self.script_path}")
            self.process = subprocess.Popen(
                [sys.executable, self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=False
            )

            # Create threads to handle stdout and stderr
            stdout_thread = Thread(
                target=self.handle_output, 
                args=(self.process.stdout,),
                daemon=True
            )
            stderr_thread = Thread(
                target=self.handle_output, 
                args=(self.process.stderr, True),
                daemon=True
            )

            stdout_thread.start()
            stderr_thread.start()

            self.monitor_script()

        except Exception as e:
            logging.error(f"Error starting script: {e}")
            self.restart_script()

    def monitor_script(self):
        """Monitor the script and restart if it stops"""
        while self.should_run:
            if self.process.poll() is not None:  # Process has stopped
                # Get any remaining output
                while not self.log_queue.empty():
                    line = self.log_queue.get_nowait()
                    logging.info(f"Final output: {line}")
                
                logging.warning("Script has stopped. Restarting...")
                self.restart_script()
            time.sleep(5)  # Check every 5 seconds

    def restart_script(self):
        """Restart the script"""
        if self.process:
            try:
                self.process.terminate()
                # Wait for any remaining output
                self.process.stdout.close()
                self.process.stderr.close()
                self.process.wait(timeout=5)
            except:
                pass
        self.start_script()

    def stop(self):
        """Stop the script and monitoring"""
        self.should_run = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

def main():
    # Replace with the path to your script
    SCRIPT_PATH = "makri/makri/media/scripts/radio/scrape_songs.py"
    
    if not os.path.exists(SCRIPT_PATH):
        logging.error(f"Script not found: {SCRIPT_PATH}")
        return

    runner = ScriptRunner(SCRIPT_PATH)
    
    def job():
        logging.info("Starting scheduled job")
        runner.start_script()

    # Schedule the script to run at 7:30 AM
    schedule.every().day.at("07:30").do(job)
    
    logging.info("Scheduler started. Waiting for 7:30 AM...")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping scheduler...")
        runner.stop()
        logging.info("Scheduler stopped")

if __name__ == "__main__":
    main()