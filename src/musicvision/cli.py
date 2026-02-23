"""
CLI entry point for MusicVision.

Usage:
    musicvision create <directory> --name "My Video"
    musicvision serve <directory> [--port 8000]
    musicvision info <directory>

Environment variables (LLM backend, API keys, etc.) are loaded from a .env
file in the current working directory if one exists.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
import uvicorn


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_create(args: argparse.Namespace) -> None:
    from musicvision.project import ProjectService

    project_dir = Path(args.directory).resolve()
    svc = ProjectService.create(project_dir, name=args.name)
    print(f"✓ Created project '{svc.config.name}' at {project_dir}")


def cmd_serve(args: argparse.Namespace) -> None:
    from musicvision.api.app import app, mount_project_files
    from musicvision.api.app import _project  # noqa: F811
    from musicvision.project import ProjectService

    project_dir = Path(args.directory).resolve()

    # Pre-load the project into the app's global state
    import musicvision.api.app as api_module

    api_module._project = ProjectService.open(project_dir)
    mount_project_files(project_dir)

    print(f"✓ Serving project '{api_module._project.config.name}'")
    print(f"  API:      http://localhost:{args.port}/docs")
    print(f"  Project:  {project_dir}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


def cmd_info(args: argparse.Namespace) -> None:
    from musicvision.project import ProjectService

    project_dir = Path(args.directory).resolve()
    svc = ProjectService.open(project_dir)

    print(f"Project:    {svc.config.name}")
    print(f"Directory:  {project_dir}")
    print(f"Audio:      {svc.config.song.audio_file or '(none)'}")
    print(f"BPM:        {svc.config.song.bpm or '(not detected)'}")
    print(f"Duration:   {svc.config.song.duration_seconds or '(unknown)'}s")
    print(f"Scenes:     {len(svc.scenes.scenes)}")
    print(f"HuMo:       {svc.config.humo.model_size.value} @ {svc.config.humo.resolution.value}")
    print(f"FLUX:       {svc.config.flux.model.value}, {svc.config.flux.steps} steps")

    if svc.scenes.scenes:
        approved_img = sum(1 for s in svc.scenes.scenes if s.image_status == "approved")
        approved_vid = sum(1 for s in svc.scenes.scenes if s.video_status == "approved")
        print(f"  Images:   {approved_img}/{len(svc.scenes.scenes)} approved")
        print(f"  Videos:   {approved_vid}/{len(svc.scenes.scenes)} approved")


def main() -> None:
    load_dotenv()  # load .env from cwd before any env vars are read

    parser = argparse.ArgumentParser(prog="musicvision", description="AI music video production pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # create
    p_create = sub.add_parser("create", help="Create a new project")
    p_create.add_argument("directory", help="Project directory path")
    p_create.add_argument("--name", default="Untitled Project", help="Project name")

    # serve
    p_serve = sub.add_parser("serve", help="Start the API server for a project")
    p_serve.add_argument("directory", help="Project directory path")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to serve on")

    # info
    p_info = sub.add_parser("info", help="Show project info")
    p_info.add_argument("directory", help="Project directory path")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "create":
        cmd_create(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
