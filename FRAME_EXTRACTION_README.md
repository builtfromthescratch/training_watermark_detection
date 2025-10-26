# Frame Extraction for YOLO Training

This folder contains scripts to extract frames from videos for watermark detection training.

## 📁 Files

- **`extract_frames.py`** - Python script that extracts frames from videos
- **`frame_counter.txt`** - Tracks which video to process next
- **`.github/workflows/frame_extraction_scheduler.yml`** - GitHub Actions workflow for automated extraction

## 🚀 How It Works

### Local Usage

1. **Place videos** in the `VIDEOS/` folder named as `Video_1.mp4`, `Video_2.mp4`, etc.

2. **Run the script**:
   ```bash
   python extract_frames.py
   ```

3. **Extracted frames** will be saved in `IMAGES/` folder as:
   - `Video_1_frame_00000.jpg`
   - `Video_1_frame_00030.jpg`
   - etc.

4. **Batch processing**: The script processes 10 videos at a time and updates `frame_counter.txt` automatically.

### GitHub Actions (Automated)

The workflow runs automatically every 6 hours (or manually trigger it):

1. **Reads** `frame_counter.txt` to know which videos to process
2. **Checks out** only the needed videos (sparse checkout)
3. **Extracts frames** from 10 videos
4. **Commits** extracted frames to the `IMAGES/` folder
5. **Updates** `frame_counter.txt` for the next batch

## ⚙️ Configuration

### Change Batch Size

Edit `extract_frames.py`:
```python
videos_per_batch = 10  # Change this number
```

### Change Frame Interval

Edit `extract_frames.py`:
```python
frame_interval = 30  # Extract every 30th frame (change as needed)
```

### Change Schedule

Edit `.github/workflows/frame_extraction_scheduler.yml`:
```yaml
schedule:
  - cron: '0 */6 * * *'  # Every 6 hours
  # - cron: '0 0 * * *'  # Daily at midnight
  # - cron: '0 */3 * * *'  # Every 3 hours
```

## 📊 Progress Tracking

The script automatically:
- ✅ Detects and crops embedded video areas (removes black borders)
- ✅ Extracts frames at specified intervals
- ✅ Tracks progress with `frame_counter.txt`
- ✅ Logs extraction statistics

### Example Output

```
Processing Video_1.mp4 (1/10)
Detected video area for Video_1: 1920x1080 at (100, 50)
  Processing Video_1: 100/3000 frames
✅ Video_1: Extracted 100 frames from 3000 total frames

============================================================
BATCH COMPLETE!
Processed: 10 videos
Total frames extracted: 985
Frames saved to: IMAGES/
============================================================
```

## 🎯 Next Steps

After extracting frames:

1. **Review** extracted frames in `IMAGES/` folder
2. **Use** `Train_Watermark_Detector.ipynb` to:
   - Auto-label frames
   - Train YOLO model
   - Generate `best.pt` model file

## 💡 Tips

- **Start small**: Test with 10 videos first
- **Monitor**: Check GitHub Actions logs for progress
- **Storage**: Each video extracts ~50-100 frames (~10-20MB)
- **Resume**: If interrupted, just run again - it picks up where it left off

## 🐛 Troubleshooting

**Issue**: `frame_counter.txt` not found
- **Solution**: The script creates it automatically, starting from Video_1

**Issue**: No frames extracted
- **Solution**: Check if videos are named correctly (`Video_1.mp4`, `Video_2.mp4`, etc.)

**Issue**: Black borders in extracted frames
- **Solution**: The script auto-detects and crops them. Check logs for detection info.

---

**Total Videos**: 790 (Video_1.mp4 to Video_790.mp4)  
**Estimated Batches**: 79 batches (10 videos each)  
**Estimated Total Frames**: ~40,000-80,000 frames
