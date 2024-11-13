from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import requests
import time
from datetime import datetime
import os
import logging
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

class RadioSongScraper:
    def __init__(self, target_artist="Taylor Swift", consecutive_count=2):
        print("Initializing RadioSongScraper...")
        # Configuration
        self.target_artist = target_artist
        self.consecutive_count = consecutive_count
        self.url = "https://www.star959.ca/play/"
        
        # Driver settings
        self.driver = None
        self.previous_songs = []  # List to store multiple previous songs
        self.last_check_time = None
        self.page_load_timeout = 10
        self.element_timeout = 5
        
        # Telegram configuration
        self.telegram_config = {
            'bot_token': 'YOUR BOT TOKEN HERE',
            'chat_id': 'YOUR CHAT ID HERE'
        }
        
        self.setup_logging()
        self.initialize_driver()
        
        self.logger.info(f"Configured to monitor for {consecutive_count} consecutive songs by {target_artist}")

    def setup_logging(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        log_filename = f"logs/song_monitor_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging

    def initialize_driver(self):
        start_time = time.time()
        self.logger.info("Initializing Chrome driver...")
        
        try:
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
            
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-logging')
            chrome_options.add_argument('--ignore-certificate-errors')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(self.page_load_timeout)
            self.wait = WebDriverWait(self.driver, self.element_timeout)
            
            initialization_time = time.time() - start_time
            self.logger.info(f"Chrome driver initialized successfully in {initialization_time:.2f} seconds")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing Chrome driver: {e}")
            return False

    def perform_health_check(self):
        """Perform a health check of the driver and connection."""
        try:
            start_time = time.time()
            self.driver.current_url
            
            try:
                self.driver.find_element(By.TAG_NAME, "body")
                elapsed = time.time() - start_time
                if elapsed > 5:
                    self.logger.warning(f"Health check slow: {elapsed:.2f} seconds")
                    return False
                return True
            except:
                return False
        except:
            return False

    def force_refresh(self):
        """Force a complete refresh of the driver and page."""
        try:
            self.logger.info("Performing force refresh...")
            self.initialize_driver()
            self.driver.get(self.url)
            time.sleep(2)
            return True
        except Exception as e:
            self.logger.error(f"Force refresh failed: {e}")
            return False

    def check_driver_connection(self):
        try:
            _ = self.driver.current_url
            return True
        except:
            return False

    def get_current_song(self):
        start_time = time.time()
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not self.check_driver_connection():
                    self.logger.info("Driver disconnected. Reinitializing...")
                    if not self.initialize_driver():
                        raise Exception("Failed to reinitialize driver")
                
                if not self.last_check_time or (datetime.now() - self.last_check_time).seconds >= 10:
                    self.logger.info(f"\nReloading page: {self.url}")
                    self.driver.get(self.url)
                    time.sleep(2)
                    self.last_check_time = datetime.now()
                else:
                    self.driver.refresh()
                    time.sleep(1)

                song_info = {
                    'time': datetime.now().strftime('%I:%M %p'),
                    'title': 'N/A',
                    'artist': 'N/A'
                }

                songs = self.driver.find_elements(By.CSS_SELECTOR, ".recently-played li")
                if songs:
                    title = songs[0].find_element(By.CSS_SELECTOR, ".song-title").text.strip()
                    artist = songs[0].find_element(By.CSS_SELECTOR, ".artist").text.strip()
                    if title and artist:
                        song_info = {
                            'time': datetime.now().strftime('%I:%M %p'),
                            'title': title,
                            'artist': artist,
                            'key': f"{title}-{artist}"
                        }
                        self.logger.info(f"Found current song: {title} by {artist}")
                        return song_info

            except WebDriverException as e:
                self.logger.error(f"WebDriver error (attempt {retry_count + 1} of {max_retries}): {e}")
                retry_count += 1
                if retry_count < max_retries:
                    self.logger.info("Attempting to recover...")
                    time.sleep(5)
                    self.initialize_driver()
                else:
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                raise

        return None

    def is_target_artist_song(self, song):
        """Check if a song is by the target artist, including features."""
        return self.target_artist.lower() in song['artist'].lower()

    def check_consecutive_songs(self):
        """Check if we have reached the desired number of consecutive songs."""
        if len(self.previous_songs) >= self.consecutive_count:
            recent_songs = self.previous_songs[-self.consecutive_count:]
            all_by_target = all(self.is_target_artist_song(song) for song in recent_songs)
            all_different = len(set(song['key'] for song in recent_songs)) == len(recent_songs)
            return all_by_target and all_different
        return False

    def send_telegram_message(self, message):
        try:
            url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"
            params = {
                'chat_id': self.telegram_config['chat_id'],
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, params=params)
            if response.status_code == 200:
                self.logger.info("Telegram notification sent successfully")
                return True
            else:
                self.logger.error(f"Error sending Telegram message: {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False

    def send_notification(self, current_song, is_consecutive=False):
        try:
            if is_consecutive:
                consecutive_songs = self.previous_songs[-(self.consecutive_count):]
                message = f"🎵 {self.consecutive_count} Consecutive {self.target_artist} Songs! 🎵\n\n"
                for i, song in enumerate(consecutive_songs, 1):
                    message += f"{i}. {song['title']} at {song['time']}\n"
            else:
                message = (f"🎵 {self.target_artist} Alert! 🎵\n\n"
                          f"Now Playing on Star 95.9:\n"
                          f"Song: {current_song['title']}\n"
                          f"Artist: {current_song['artist']}\n"
                          f"Time: {current_song['time']}\n\n"
                          f"Watching for more {self.target_artist} songs...")
            
            if self.send_telegram_message(message):
                self.logger.info("Notification sent successfully")
            else:
                self.logger.error("Failed to send notification")

        except Exception as e:
            self.logger.error(f"Error in send_notification: {e}")

    def monitor_playlist(self, check_interval=30):
        self.logger.info(f"Monitoring playlist for {self.target_artist} songs. "
                        f"Looking for {self.consecutive_count} consecutive songs. "
                        f"Checking every {check_interval} seconds...")
        
        notification_cooldown = 60
        last_notification_time = None
        consecutive_errors = 0
        max_consecutive_errors = 3
        last_health_check = time.time()
        health_check_interval = 300
        
        while True:
            try:
                current_time = time.time()
                
                if current_time - last_health_check > health_check_interval:
                    self.logger.info("Performing periodic health check...")
                    if not self.perform_health_check():
                        self.logger.warning("Health check failed - forcing refresh")
                        self.force_refresh()
                    last_health_check = current_time
                
                current = self.get_current_song()
                if current and current['artist'] != 'N/A':
                    consecutive_errors = 0
                    current_datetime = datetime.now()
                    
                    # Check if it's a target artist song
                    if self.is_target_artist_song(current):
                        if not last_notification_time or (current_datetime - last_notification_time).seconds >= notification_cooldown:
                            # Add to previous songs if it's different
                            if not self.previous_songs or current['key'] != self.previous_songs[-1]['key']:
                                self.previous_songs.append(current)
                                # Keep only recent history
                                self.previous_songs = self.previous_songs[-self.consecutive_count:]
                                
                                if self.check_consecutive_songs():
                                    self.logger.info(f"\nDetected {self.consecutive_count} consecutive {self.target_artist} songs!")
                                    self.send_notification(current, is_consecutive=True)
                                else:
                                    self.logger.info(f"\nDetected {self.target_artist} song!")
                                    self.send_notification(current, is_consecutive=False)
                                
                                last_notification_time = current_datetime
                    
                    self.logger.info(f"\nCurrent song at {current['time']}:")
                    self.logger.info(f"Title: {current['title']}")
                    self.logger.info(f"Artist: {current['artist']}")
                    self.logger.info("-" * 50)
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                self.logger.info("\nStopping monitor...")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error("Too many consecutive errors. Restarting driver...")
                    self.initialize_driver()
                    consecutive_errors = 0
                time.sleep(check_interval)

    def cleanup(self):
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()
                self.logger.info("Browser closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing browser: {e}")

def main():
    # Configuration
    target_artist = "Taylor Swift"  # Change this to monitor a different artist
    consecutive_count = 2           # Change this to monitor for more consecutive songs
    check_interval = 30            # Seconds between checks
    
    scraper = None
    try:
        print(f"Starting song scraper for {target_artist}...")
        scraper = RadioSongScraper(target_artist=target_artist, consecutive_count=consecutive_count)
        
        print(f"Starting playlist monitoring (checking every {check_interval} seconds)...")
        scraper.monitor_playlist(check_interval=check_interval)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if scraper:
            scraper.cleanup()

if __name__ == "__main__":
    main()