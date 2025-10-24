import subprocess
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# === CONFIGURATION ===
scripts_to_run = ["Task1.py", "Task1-manual.py"]  # your two scripts
output_dir = Path("confusion_matrix_outputs")
output_dir.mkdir(exist_ok=True)

# Create subfolders for each script
script_folders = [output_dir / f"script_{i+1}" for i in range(len(scripts_to_run))]
for folder in script_folders:
    folder.mkdir(exist_ok=True)

# Function to save current figure uniquely in the correct folder
def save_confusion_matrix_plot(folder, plot_title="confusion_matrix"):
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = f"{plot_title}_{timestamp}.png"
    filepath = folder / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath

# === STEP 1: Run the scripts ===
# NOTE: Each script must call `save_confusion_matrix_plot(folder, title)` after plotting
for i, script in enumerate(scripts_to_run):
    print(f"Running {script} ...")
    subprocess.run(["python3", script], check=True)

# === STEP 2: Collect saved images ===
all_images = [sorted(folder.iterdir()) for folder in script_folders]

# Determine max number of images among scripts
max_images = max(len(img_list) for img_list in all_images)

# === STEP 3: Write Markdown file in paired Obsidian format ===
md_path = output_dir / "confusion_matrices.md"
with open(md_path, "w", encoding="utf-8") as f:
    for img_idx in range(max_images):
        f.write("```image-layout-a\n")
        for script_idx, img_list in enumerate(all_images):
            if img_idx < len(img_list):
                img_file = img_list[img_idx].name
                # Obsidian-style image link
                f.write(f"![[{img_file}]]\n")
        f.write("```\n")

print(f"\n✅ Images saved in {output_dir}")
print(f"✅ Markdown file generated: {md_path}")
