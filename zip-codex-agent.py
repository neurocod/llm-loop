"""Archive this directory's agent folders, verify the ZIP, then remove them."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys
from contextlib import contextmanager
from datetime import datetime
from zipfile import ZIP_DEFLATED, ZipFile


@contextmanager
def pending_archive(archive: Path):
    if archive.exists():
        raise FileExistsError(f"Archive already exists: {archive}")
    partial = archive.with_suffix(".zip.partial")
    # Reserve exclusively so a failed run cannot remove another run's file.
    with partial.open("xb"):
        pass
    try:
        yield partial
    finally:
        partial.unlink(missing_ok=True)


def digest(stream) -> bytes:
    result = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        result.update(chunk)
    return result.digest()


def inventory(root: Path, folders: list[Path]) -> tuple[list[Path], list[Path]]:
    def fail(error: OSError) -> None:
        raise error

    files = []
    directories = []
    for folder in folders:
        for current, children, names in os.walk(folder, followlinks=False, onerror=fail):
            directory = Path(current)
            directories.append(directory)
            for path in [directory, *(directory / name for name in children + names)]:
                attributes = getattr(path.lstat(), "st_file_attributes", 0)
                if path.is_symlink() or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    raise RuntimeError(f"Refusing to archive a link or junction: {path}")
            files.extend(directory / name for name in names)
    return sorted(files), sorted(directories)


def main(root: Path) -> int:
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"State directory does not exist: {root}")
    print("WARNING: Run this script ONLY when there are no active Codex sessions.")
    print("This script cannot detect active sessions. Stop them before continuing.")
    print(f"All immediate subfolders of {root} will be archived and deleted.")
    print("registry.json will be archived and deleted after successful cleanup.")
    if input("Type ARCHIVE to continue: ").strip() != "ARCHIVE":
        print("Cancelled. No changes made.")
        return 0

    folders = sorted(path for path in root.iterdir() if path.is_dir())
    registry = root / "registry.json"
    if not folders and not registry.exists():
        print("No folders or registry to archive.")
        return 0
    files, directories = inventory(root, folders)
    if registry.exists():
        if registry.is_symlink():
            raise RuntimeError(f"Refusing to archive a link: {registry}")
        files.append(registry)
    archive = root / f"{datetime.now():%Y.%m.%d-%H.%M.%S}-{len(folders)}.zip"
    hashes = {}
    members = {folder.name: [] for folder in folders}
    for path in directories + [path for path in files if path != registry]:
        members[path.relative_to(root).parts[0]].append(path)
    # Exclusive creation prevents overwriting an earlier archive. Until every
    # member has been checked against its source, all source folders stay intact.
    print(f"Archiving {len(folders)} folders: ", end="", flush=True)
    with pending_archive(archive) as partial:
        with ZipFile(partial, "w", compression=ZIP_DEFLATED, allowZip64=True, strict_timestamps=False) as output:
            for folder in folders:
                for path in members[folder.name]:
                    if path.is_dir():
                        output.write(path, path.relative_to(root).as_posix() + "/")
                    else:
                        with path.open("rb") as source:
                            hashes[path] = digest(source)
                        output.write(path, path.relative_to(root).as_posix())
                print(".", end="", flush=True)
            if registry in files:
                path = registry
                with path.open("rb") as source:
                    hashes[path] = digest(source)
                output.write(path, path.relative_to(root).as_posix())
        print("\nVerifying archive and source files...", flush=True)

        with ZipFile(partial) as saved:
            for path, expected in hashes.items():
                with saved.open(path.relative_to(root).as_posix()) as member:
                    if digest(member) != expected:
                        raise RuntimeError(f"Archive verification failed: {path}")
                with path.open("rb") as source:
                    if digest(source) != expected:
                        raise RuntimeError(f"Source changed during archiving: {path}")
        current_files, current_directories = inventory(root, folders)
        if current_files != sorted(path for path in files if path != registry) or current_directories != directories:
            raise RuntimeError("Folder contents changed during archiving; nothing deleted.")
        # A hard link publishes the verified file atomically without replacing
        # an existing archive, even if another invocation picked the same name.
        os.link(partial, archive)

    print(f"Verified archive: {archive}")
    for folder in folders:
        # Only the immediate folders inventoried under the script's own directory
        # may be deleted. Running from another working directory cannot widen it.
        if folder.parent != root or folder.resolve() != folder:
            raise RuntimeError(f"Refusing to delete an unexpected path: {folder}")
        shutil.rmtree(folder)
    if registry in hashes:
        registry.unlink()
    registry_message = " and registry.json" if registry in hashes else ""
    print(f"Removed {len(folders)} folders{registry_message}. Archive retained.")
    return 0


def cli(root: Path | None = None) -> int:
    try:
        if root is None:
            parser = argparse.ArgumentParser(description=__doc__)
            parser.add_argument("--state-dir", type=Path, default=Path.cwd() / ".codex-agents")
            root = parser.parse_args().state_dir
        return main(root)
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 1
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
