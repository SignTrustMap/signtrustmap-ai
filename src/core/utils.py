import os

def get_reload_excludes() -> list[str]:
    excludes = ["logs"]
    if os.path.exists(".gitignore"):
        with open(".gitignore", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("/"):
                    line = line[1:]
                
                if line.endswith("/"):
                    excludes.append(line[:-1])
                elif "." not in line and "*" not in line:
                    excludes.append(line)
                elif line.startswith("*."):
                    excludes.append(line)
    return list(set(excludes))
