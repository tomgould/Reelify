from pathlib import Path

from scenedetect import open_video, SceneManager, ContentDetector
from scenedetect.scene_manager import save_images


def extract_keyframes(video_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    video = open_video(str(video_path), backend="opencv")
    manager = SceneManager()
    manager.add_detector(ContentDetector())
    manager.detect_scenes(video)
    scene_list = manager.get_scene_list()

    if scene_list:
        save_images(scene_list, video, num_images=1, output_dir=str(output_dir))

    return sorted(output_dir.glob("*.jpg"))
