import hashlib
import json


def artifact_digest(directory, patterns):
    """Hash matching artifact contents and their relative paths."""
    files = sorted(path for path in directory.rglob("*") if path.is_file()
        and any(path.match(pattern) for pattern in patterns) and ".cache" not in path.parts)
    if not files:
        raise ValueError("model or source directory has no expected artifacts: " + str(directory))
    entries = []
    for path in files:
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append([path.relative_to(directory).as_posix(), checksum])
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
