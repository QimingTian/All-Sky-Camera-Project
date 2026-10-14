# ASC Label Tool

Points at `/Volumes/James' SSD/ASC Training Data` (images stay flat there).
Labels are saved here as `labels.csv`.

```bash
cd "/Volumes/James' SSD/ASC Label Tool"
python3 label_server.py
```

Opens http://127.0.0.1:8765/

| Field | Values | Keys |
|-------|--------|------|
| sky | clear / cloudy | 1 / 2 |
| rain | true / false | R / F |
| snow | true / false | S / Q |
| day_night | day / night | Y / N (prefilled from filename) |
