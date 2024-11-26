import os

def manage_gitkeep(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        
        gitkeep_path = os.path.join(dirpath, ".gitkeep")
        files_without_gitkeep = [f for f in filenames if f != '.gitkeep']

        if not files_without_gitkeep:
            if not os.path.exists(gitkeep_path):
                with open(gitkeep_path, 'w') as f:
                    f.write("")
                print(f"Added .gitkeep to: {dirpath}")
        else:
            if os.path.exists(gitkeep_path):
                os.remove(gitkeep_path)
                print(f"Removed .gitkeep from: {dirpath}")

repo_path = r'F:\Projects\Hand-Gesture-Reader'
manage_gitkeep(repo_path)
