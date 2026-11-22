import time
from datetime import datetime
import csv
import os
import config

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

class AttendanceManager:
    """
    Manages live real-time attendance in memory (no external database needed).
    Tracks live entries with timestamp, cooldown duplicate prevention,
    audio alert feedback, and optional CSV export.
    """
    def __init__(self, cooldown_seconds=config.ATTENDANCE_COOLDOWN_SEC):
        self.cooldown_seconds = cooldown_seconds
        self.records = []                   # In-memory list of attendance dicts
        self.marked_tracker = {}            # {roll_no: {'last_timestamp': float, 'time_str': str}}
        self.listeners = []                 # Callbacks for live UI updates

    def register_listener(self, callback):
        """Registers a callback function to be called when new attendance is marked."""
        self.listeners.append(callback)

    def is_already_marked(self, roll_no):
        """Checks if the student was marked within the cooldown window."""
        roll_no = str(roll_no)
        if roll_no not in self.marked_tracker:
            return False
        elapsed = time.time() - self.marked_tracker[roll_no]["last_timestamp"]
        return elapsed < self.cooldown_seconds

    def get_last_marked_time(self, roll_no):
        roll_no = str(roll_no)
        if roll_no in self.marked_tracker:
            return self.marked_tracker[roll_no]["time_str"]
        return None

    def mark_attendance(self, roll_no, name, confidence):
        """
        Marks attendance for a detected student.
        Returns:
            is_newly_marked (bool): True if marked just now, False if already marked recently
            status_message (str): e.g. "Present", "Already Marked at 12:35 PM"
            record (dict or None): The newly created record
        """
        roll_no = str(roll_no)
        now = datetime.now()
        current_time_str = now.strftime("%I:%M:%S %p")
        current_date_str = now.strftime("%Y-%m-%d")

        if self.is_already_marked(roll_no):
            last_time = self.marked_tracker[roll_no]["time_str"]
            return False, f"Marked ({last_time})", None

        # Create new record
        record = {
            "index": len(self.records) + 1,
            "roll_no": roll_no,
            "name": name,
            "date": current_date_str,
            "time": current_time_str,
            "confidence": f"{confidence * 100:.1f}%",
            "status": "Present"
        }

        self.records.append(record)
        self.marked_tracker[roll_no] = {
            "last_timestamp": time.time(),
            "time_str": current_time_str
        }

        # Play quick success confirmation beep
        if HAS_WINSOUND:
            try:
                winsound.Beep(1200, 180)
            except Exception:
                pass

        # Notify any UI listeners
        for listener in self.listeners:
            try:
                listener(record)
            except Exception as e:
                print(f"Listener error: {e}")

        return True, "Present", record

    def get_records(self):
        return list(self.records)

    def get_total_present(self):
        return len(self.records)

    def clear_session(self):
        """Resets the live attendance session."""
        self.records.clear()
        self.marked_tracker.clear()

    def export_to_csv(self, filepath=None):
        """Exports in-memory live attendance records to a CSV file."""
        path = filepath or config.DEFAULT_CSV_PATH
        if not self.records:
            return False, "No attendance records to export."

        try:
            with open(path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Sr #", "Roll No", "Student Name", "Date", "Time", "Confidence", "Status"])
                for r in self.records:
                    writer.writerow([r["index"], r["roll_no"], r["name"], r["date"], r["time"], r["confidence"], r["status"]])
            return True, f"Exported {len(self.records)} records to {os.path.basename(path)}"
        except Exception as e:
            return False, f"Export error: {e}"

