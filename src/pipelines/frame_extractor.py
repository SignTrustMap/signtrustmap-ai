import os
import cv2
from loguru import logger


def extract_video_frame(video_path, frame_index, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file to extract frame: {video_path}")
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if ret:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if success:
            logger.debug(f"Extracted video frame {frame_index} to: {output_path}")
        else:
            logger.error(f"Failed to write extracted frame to: {output_path}")
        return success
    logger.error(f"Failed to read frame {frame_index} from video: {video_path}")
    return False


def extract_video_frame_at_time(video_path, time_seconds, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file to extract frame: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_index = int(time_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if ret:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if success:
            logger.debug(f"Extracted video frame at {time_seconds}s (index {frame_index}) to: {output_path}")
        else:
            logger.error(f"Failed to write extracted frame to: {output_path}")
        return success
    logger.error(f"Failed to read frame at {time_seconds}s (index {frame_index}) from video: {video_path}")
    return False
