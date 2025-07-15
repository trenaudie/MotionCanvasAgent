import re
from pathlib import Path

AGENTS_PROMPTS_DIR = Path( 'agents')/ 'prompts'
SCENES_DIR = Path( 'frontend' )/ 'src' / 'scenes'
OUTPUT_FILE = SCENES_DIR / "example.tsx"
PROJECT_ROOT = Path(".")

def get_next_scene_filepath() -> Path:
    numbered_files = []
    pattern = re.compile(r"^(\d+)_scene\.tsx$")

    for file in SCENES_DIR.glob("*.tsx"):
        match = pattern.match(file.name)
        if match:
            numbered_files.append(int(match.group(1)))

    if not numbered_files:
        next_number = 1
    else:
        next_number = max(numbered_files) + 1

    return SCENES_DIR / f"{next_number:02d}_scene.tsx"


def clear_scenes_dir(after_idx: int) -> None:
    """
    Remove all numbered scene files in SCENES_DIR with index > after_idx.
    Keeps files like 'dummy.tsx' or '03_scene.tsx' if 03 <= after_idx.
    """
    pattern = re.compile(r"^(\d+)_scene\.tsx$")

    for file in SCENES_DIR.glob("*.tsx"):
        match = pattern.match(file.name)
        if match:
            file_idx = int(match.group(1))
            if file_idx > after_idx:
                print(f"Deleting {file.name}")
                file.unlink()