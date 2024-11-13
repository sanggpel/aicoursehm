import unittest
from datetime import datetime
import time
from swift_alert import RadioSongScraper

class TestNotifications:
    def __init__(self, target_artist="Taylor Swift", consecutive_count=2):
        # Initialize the scraper with test configuration
        self.scraper = RadioSongScraper(target_artist=target_artist, consecutive_count=consecutive_count)
        self.target_artist = target_artist
        self.consecutive_count = consecutive_count
        self.setup_test_data()

    def setup_test_data(self):
        """Setup test song data."""
        current_time = datetime.now().strftime('%I:%M %p')
        
        # Setup single song
        self.single_song = {
            'title': 'Shake It Off',
            'artist': self.target_artist,
            'time': current_time,
            'key': f'Shake It Off-{self.target_artist}'
        }
        
        # Setup consecutive songs
        self.test_songs = []
        test_titles = ['Shake It Off', 'Anti-Hero', 'Cruel Summer', 'Love Story']  # Add more if needed
        
        for i in range(self.consecutive_count):
            self.test_songs.append({
                'title': test_titles[i],
                'artist': self.target_artist,
                'time': current_time,
                'key': f'{test_titles[i]}-{self.target_artist}'
            })

    def verify_telegram_credentials(self):
        """Verify Telegram credentials before running tests."""
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.scraper.telegram_config['bot_token']}/getMe"
            response = requests.get(url)
            if response.status_code == 200:
                print("✓ Telegram credentials verified successfully")
                return True
            else:
                print(f"✗ Telegram credential verification failed: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Telegram credential verification failed: {str(e)}")
            return False

    def test_single_song_notification(self):
        """Test notification for a single song."""
        print(f"\nTesting Single {self.target_artist} Song Notification:")
        print("-" * 50)
        success = True
        error_messages = []
        
        try:
            self.scraper.send_notification(self.single_song, is_consecutive=False)
        except Exception as e:
            success = False
            error_messages.append(str(e))
            
        if success:
            print("✓ Single song notification test completed successfully")
        else:
            print("✗ Single song notification test failed:")
            for error in error_messages:
                print(f"  - {error}")
            
        return success

    def test_consecutive_song_notification(self):
        """Test notification for consecutive songs."""
        print(f"\nTesting {self.consecutive_count} Consecutive Songs Notification:")
        print("-" * 50)
        success = True
        error_messages = []
        
        try:
            # Set up previous songs
            self.scraper.previous_songs = self.test_songs[:-1]  # All but the last song
            # Send notification with the last song
            self.scraper.send_notification(self.test_songs[-1], is_consecutive=True)
        except Exception as e:
            success = False
            error_messages.append(str(e))
            
        if success:
            print(f"✓ {self.consecutive_count} consecutive songs notification test completed successfully")
        else:
            print("✗ Consecutive songs notification test failed:")
            for error in error_messages:
                print(f"  - {error}")
            
        return success

    def test_artist_detection(self):
        """Test artist detection logic."""
        print("\nTesting Artist Detection:")
        print("-" * 50)
        
        test_cases = [
            {'artist': self.target_artist, 'title': 'Test Song'},
            {'artist': self.target_artist.upper(), 'title': 'Test Song'},
            {'artist': self.target_artist.lower(), 'title': 'Test Song'},
            {'artist': f'{self.target_artist} f/Someone', 'title': 'Test Song'},
            {'artist': 'Other Artist', 'title': 'Test Song'}
        ]
        
        for song in test_cases:
            result = self.scraper.is_target_artist_song(song)
            expected = self.target_artist.lower() in song['artist'].lower()
            print(f"Testing artist '{song['artist']}': {'✓' if result == expected else '✗'}")

def main():
    # Test configuration
    target_artist = "Taylor Swift"  # Change to test different artist
    consecutive_count = 2          # Change to test different number of consecutive songs
    
    print(f"Starting Notification Tests for {target_artist}")
    print("=" * 50)
    print(f"Configured to test {consecutive_count} consecutive songs")
    
    tester = TestNotifications(target_artist=target_artist, consecutive_count=consecutive_count)
    
    # Verify credentials first
    print("\nVerifying credentials:")
    print("-" * 50)
    telegram_creds_ok = tester.verify_telegram_credentials()
    
    if not telegram_creds_ok:
        print("\nTelegram credential verification failed. Please check your settings.")
        print("\nTest aborted.")
        return
    
    print("\nCredentials verified successfully.")
    
    # Test artist detection
    tester.test_artist_detection()
    
    # Test notifications
    proceed = input("\nProceed with sending test notifications? (y/n): ")
    
    if proceed.lower() == 'y':
        # Test single song notification
        single_result = tester.test_single_song_notification()
        
        if single_result:
            print(f"\nWaiting 5 seconds before testing {consecutive_count} consecutive songs...")
            time.sleep(5)
            
            # Test consecutive songs notification
            consecutive_result = tester.test_consecutive_song_notification()
        else:
            consecutive_result = False
            print("\nSkipping consecutive test due to single notification failure")
        
        print("\nTest Results Summary:")
        print("=" * 50)
        print(f"Single Song Notification: {'✓ Success' if single_result else '✗ Failed'}")
        print(f"Consecutive Songs Notification: {'✓ Success' if consecutive_result else '✗ Failed'}")
        
        if single_result and consecutive_result:
            print("\nAll tests passed successfully!")
        else:
            print("\nSome tests failed. Please check the error messages above.")
    else:
        print("\nTests aborted by user.")

if __name__ == "__main__":
    main()