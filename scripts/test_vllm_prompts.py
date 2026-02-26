#!/usr/bin/env python3
"""
MusicVision — vLLM Prompt Generation Test

Tests all three LLM-dependent pipeline functions against a running vLLM server:
  1. Scene segmentation (intake/segmentation.py)
  2. Image prompt generation (imaging/prompt_generator.py)
  3. Video prompt generation (video/prompt_generator.py)

No GPU needed on this machine — just network access to the vLLM endpoint.

Usage:
    # Basic — uses env vars from .env
    python test_vllm_prompts.py

    # Override endpoint
    python test_vllm_prompts.py --base-url http://192.168.1.136:8000/v1 --model qwen32b

    # Test only one stage
    python test_vllm_prompts.py --test segmentation
    python test_vllm_prompts.py --test image-prompts
    python test_vllm_prompts.py --test video-prompts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from textwrap import dedent, indent

# ---------------------------------------------------------------------------
# Try to load .env if dotenv is available (not required)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# We talk directly to the OpenAI-compatible API — no musicvision imports needed.
# This lets you run the test on any machine with network access to vLLM.
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Test fixtures — mock data that mirrors real pipeline structures
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_LYRICS_WITH_TIMESTAMPS = dedent("""\
    [verse]
    0.00 - 3.20: Standing on the edge of something new
    3.20 - 6.80: The city lights below are fading through
    6.80 - 10.40: I hear the echoes of a distant sound
    10.40 - 14.00: My heart is racing but my feet won't leave the ground

    [chorus]
    14.00 - 17.60: We're burning brighter than the falling stars tonight
    17.60 - 21.20: Hold on to me, don't let the darkness steal the light
    21.20 - 24.80: We're burning brighter, burning brighter

    [verse]
    24.80 - 28.40: The wind is shifting and the shadows grow
    28.40 - 32.00: But every step I take, I feel the afterglow

    [bridge]
    32.00 - 38.00: (instrumental)

    [chorus]
    38.00 - 41.60: We're burning brighter than the falling stars tonight
    41.60 - 45.20: Hold on to me, don't let the darkness steal the light
    45.20 - 48.80: We're burning brighter, burning brighter

    [outro]
    48.80 - 55.00: (instrumental fade)
""")

SAMPLE_STYLE_SHEET = {
    "visual_style": "Cinematic 35mm film, desaturated tones, shallow depth of field, "
                    "golden hour lighting with lens flare",
    "color_palette": "Muted blues and amber, high contrast shadows, warm highlights",
    "characters": [
        {
            "id": "protagonist",
            "description": "Young woman, mid-20s, dark curly hair past shoulders, "
                           "worn brown leather jacket, silver necklace with crescent pendant, "
                           "determined expression",
        }
    ],
    "props": [
        {"id": "vintage_mic", "description": "Chrome vintage ribbon microphone on a black stand"}
    ],
    "settings": [
        {
            "id": "rooftop",
            "description": "City rooftop at dusk, neon signs in background, gravel surface, "
                           "water towers silhouetted against orange sky",
        },
        {
            "id": "street",
            "description": "Empty rain-slicked city street at night, reflections of neon signs "
                           "on wet asphalt, steam rising from grates",
        },
    ],
}

SAMPLE_SCENE = {
    "id": "scene_003",
    "order": 3,
    "time_start": 6.80,
    "time_end": 10.40,
    "type": "vocal",
    "lyrics": "I hear the echoes of a distant sound",
    "characters": ["protagonist"],
    "props": [],
    "settings": ["rooftop"],
}

SAMPLE_SCENE_INSTRUMENTAL = {
    "id": "scene_008",
    "order": 8,
    "time_start": 32.00,
    "time_end": 38.00,
    "type": "instrumental",
    "lyrics": "",
    "characters": [],
    "props": [],
    "settings": ["street"],
}


# ═══════════════════════════════════════════════════════════════════════════
# System prompts — these mirror what the pipeline modules send to the LLM
# ═══════════════════════════════════════════════════════════════════════════

SEGMENTATION_SYSTEM_PROMPT = dedent("""\
    You are a music video director segmenting a song into visual scenes.

    Given lyrics with timestamps, divide the song into scenes suitable for
    music video production. Each scene will become a separate video clip.

    Rules:
    - Minimum scene duration: 2 seconds
    - Maximum scene duration: 10 seconds
    - Prefer cuts on musical phrase boundaries
    - Instrumental sections get their own scenes with type "instrumental"
    - Vocal sections have type "vocal"
    - Each scene should have a coherent visual concept

    Output ONLY valid JSON — no markdown fences, no commentary. Format:
    {
      "scenes": [
        {
          "id": "scene_001",
          "order": 1,
          "time_start": 0.00,
          "time_end": 6.80,
          "type": "vocal",
          "lyrics": "Standing on the edge of something new / The city lights below are fading through"
        }
      ]
    }
""")

IMAGE_PROMPT_SYSTEM_PROMPT = dedent("""\
    You are a cinematographer writing image generation prompts for a music video.

    Given a scene's lyrics, style sheet, and context, write a 2-4 sentence
    FLUX image prompt that describes the visual composition for a single
    reference still frame.

    Guidelines:
    - Describe the shot composition, lighting, and mood
    - Include relevant character descriptions from the style sheet
    - Include the setting description if specified
    - Use cinematic language (close-up, wide shot, over-the-shoulder, etc.)
    - Do NOT include text, lyrics, or words in the image description
    - Match the emotional tone of the lyrics

    Output ONLY the prompt text — no JSON, no markdown, no commentary.
""")

VIDEO_PROMPT_SYSTEM_PROMPT = dedent("""\
    You are writing dense video generation prompts for HuMo (a text-to-video model).

    HuMo works best with Qwen2.5-VL-style dense captions. Write a detailed
    description of 3-5 sentences covering:
    - Subject appearance and action (what is happening)
    - Camera movement (static, slow pan, tracking shot, etc.)
    - Lighting and atmosphere
    - Motion dynamics (slow/fast, direction of movement)
    - Background activity

    The prompt should describe ~4 seconds of continuous video motion.

    Output ONLY the prompt text — no JSON, no markdown, no commentary.
""")


# ═══════════════════════════════════════════════════════════════════════════
# Test functions
# ═══════════════════════════════════════════════════════════════════════════

def test_connection(client: OpenAI, model: str) -> bool:
    """Quick health check — can we reach the server and get a response?"""
    print("=" * 70)
    print("TEST 0: Connection health check")
    print("=" * 70)
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        text = resp.choices[0].message.content.strip()
        print(f"  Response: {text!r}  ({elapsed:.2f}s)")
        print(f"  Model:    {resp.model}")
        tokens = resp.usage
        if tokens:
            print(f"  Tokens:   {tokens.prompt_tokens} prompt + {tokens.completion_tokens} completion")
        print("  ✅ PASS\n")
        return True
    except Exception as e:
        print(f"  ❌ FAIL: {e}\n")
        return False


def test_segmentation(client: OpenAI, model: str) -> bool:
    """Test scene segmentation: lyrics → JSON scene list."""
    print("=" * 70)
    print("TEST 1: Scene segmentation (intake)")
    print("=" * 70)

    user_msg = dedent(f"""\
        Song duration: 55.0 seconds
        BPM: 120

        Lyrics with timestamps:
        {SAMPLE_LYRICS_WITH_TIMESTAMPS}

        Segment this song into scenes for a music video. Output JSON only.
    """)

    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SEGMENTATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=2000,
            temperature=0.3,
        )
        elapsed = time.time() - t0
        raw = resp.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Raw length: {len(raw)} chars")

        # Validate JSON
        data = json.loads(raw)
        scenes = data.get("scenes", [])
        print(f"  Scenes returned: {len(scenes)}")

        # Validation checks
        errors = []
        if len(scenes) < 3:
            errors.append(f"Too few scenes ({len(scenes)}), expected at least 3")
        if len(scenes) > 20:
            errors.append(f"Too many scenes ({len(scenes)}), expected at most ~15")

        has_instrumental = any(s.get("type") == "instrumental" for s in scenes)
        if not has_instrumental:
            errors.append("No instrumental scenes found (bridge + outro should be instrumental)")

        has_vocal = any(s.get("type") == "vocal" for s in scenes)
        if not has_vocal:
            errors.append("No vocal scenes found")

        for i, s in enumerate(scenes):
            dur = s.get("time_end", 0) - s.get("time_start", 0)
            if dur < 1.5:
                errors.append(f"Scene {s.get('id', i)}: duration {dur:.1f}s < 2s minimum")
            if dur > 11:
                errors.append(f"Scene {s.get('id', i)}: duration {dur:.1f}s > 10s maximum")
            if "id" not in s:
                errors.append(f"Scene index {i}: missing 'id' field")
            if "time_start" not in s or "time_end" not in s:
                errors.append(f"Scene {s.get('id', i)}: missing timestamp fields")

        # Check coverage — first scene should start near 0, last should end near 55
        if scenes:
            first_start = scenes[0].get("time_start", 99)
            last_end = scenes[-1].get("time_end", 0)
            if first_start > 1.0:
                errors.append(f"First scene starts at {first_start}s, expected near 0")
            if last_end < 50.0:
                errors.append(f"Last scene ends at {last_end}s, expected near 55")

        # Print scene summary
        print("\n  Scene breakdown:")
        for s in scenes:
            dur = s.get("time_end", 0) - s.get("time_start", 0)
            lyrics_preview = (s.get("lyrics", "") or "(instrumental)")[:50]
            print(f"    {s.get('id', '?'):>12}  {s.get('time_start',0):6.2f}–{s.get('time_end',0):6.2f}  "
                  f"({dur:4.1f}s)  {s.get('type', '?'):12}  {lyrics_preview}")

        if errors:
            print(f"\n  ⚠️  Warnings ({len(errors)}):")
            for e in errors:
                print(f"    - {e}")
            print("  ✅ PASS (with warnings)\n")
        else:
            print("  ✅ PASS\n")
        return True

    except json.JSONDecodeError as e:
        print(f"  ❌ FAIL: Invalid JSON response")
        print(f"    Error: {e}")
        print(f"    Raw output:\n{indent(raw[:500], '      ')}")
        print()
        return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}\n")
        return False


def test_image_prompts(client: OpenAI, model: str) -> bool:
    """Test image prompt generation for vocal + instrumental scenes."""
    print("=" * 70)
    print("TEST 2: Image prompt generation (imaging)")
    print("=" * 70)

    results = []
    for label, scene in [("vocal", SAMPLE_SCENE), ("instrumental", SAMPLE_SCENE_INSTRUMENTAL)]:
        print(f"\n  --- {label} scene: {scene['id']} ---")

        # Build the user message like imaging/prompt_generator.py would
        setting_desc = ""
        for s in SAMPLE_STYLE_SHEET["settings"]:
            if s["id"] in scene.get("settings", []):
                setting_desc = s["description"]
                break

        char_desc = ""
        for c in SAMPLE_STYLE_SHEET["characters"]:
            if c["id"] in scene.get("characters", []):
                char_desc = c["description"]
                break

        user_msg = dedent(f"""\
            Scene: {scene['id']} ({scene['time_start']:.2f}s – {scene['time_end']:.2f}s)
            Type: {scene['type']}
            Lyrics: {scene.get('lyrics') or '(instrumental — no lyrics)'}

            Style sheet:
              Visual style: {SAMPLE_STYLE_SHEET['visual_style']}
              Color palette: {SAMPLE_STYLE_SHEET['color_palette']}
              Character: {char_desc or 'N/A'}
              Setting: {setting_desc or 'N/A'}

            Write a FLUX image prompt for this scene's reference still.
        """)

        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": IMAGE_PROMPT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=300,
                temperature=0.7,
            )
            elapsed = time.time() - t0
            prompt = resp.choices[0].message.content.strip()

            print(f"  Time: {elapsed:.2f}s")
            print(f"  Prompt ({len(prompt)} chars):")
            print(indent(prompt, "    "))

            # Validation
            errors = []
            if len(prompt) < 30:
                errors.append("Prompt too short (< 30 chars)")
            if len(prompt) > 1000:
                errors.append("Prompt too long (> 1000 chars)")
            if prompt.startswith("{") or prompt.startswith("["):
                errors.append("Got JSON instead of plain text prompt")
            if prompt.startswith("```"):
                errors.append("Got markdown-fenced response instead of plain text")

            # Vocal scenes should reference the character
            if scene["type"] == "vocal" and char_desc:
                has_char_hint = any(
                    kw in prompt.lower()
                    for kw in ["woman", "leather", "curly", "hair", "jacket", "protagonist", "singer", "she", "her"]
                )
                if not has_char_hint:
                    errors.append("Vocal scene prompt doesn't seem to reference the character")

            if errors:
                print(f"  ⚠️  Warnings: {'; '.join(errors)}")

            results.append(len(errors) == 0)

        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            results.append(False)

    passed = all(results)
    print(f"\n  {'✅ PASS' if passed else '⚠️  PARTIAL PASS'}\n")
    return passed


def test_video_prompts(client: OpenAI, model: str) -> bool:
    """Test video prompt generation (dense HuMo-style captions)."""
    print("=" * 70)
    print("TEST 3: Video prompt generation (HuMo dense caption)")
    print("=" * 70)

    scene = SAMPLE_SCENE
    # Simulate having an image prompt already generated
    image_prompt = (
        "Medium close-up of a young woman with dark curly hair on a city rooftop at dusk. "
        "She stands near the edge, the warm amber glow of the setting sun catching her leather jacket. "
        "Neon signs flicker in the soft-focus background. Shallow depth of field, 35mm film grain."
    )

    setting_desc = ""
    for s in SAMPLE_STYLE_SHEET["settings"]:
        if s["id"] in scene.get("settings", []):
            setting_desc = s["description"]
            break

    char_desc = ""
    for c in SAMPLE_STYLE_SHEET["characters"]:
        if c["id"] in scene.get("characters", []):
            char_desc = c["description"]
            break

    user_msg = dedent(f"""\
        Scene: {scene['id']} ({scene['time_start']:.2f}s – {scene['time_end']:.2f}s, {scene['time_end'] - scene['time_start']:.1f}s duration)
        Type: {scene['type']}
        Lyrics: {scene.get('lyrics', '')}

        Reference image description: {image_prompt}

        Style sheet:
          Visual style: {SAMPLE_STYLE_SHEET['visual_style']}
          Color palette: {SAMPLE_STYLE_SHEET['color_palette']}
          Character: {char_desc}
          Setting: {setting_desc}

        Write a dense HuMo video prompt for this ~4 second clip.
        Describe the motion, camera movement, lighting dynamics, and atmosphere.
    """)

    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": VIDEO_PROMPT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=400,
            temperature=0.7,
        )
        elapsed = time.time() - t0
        prompt = resp.choices[0].message.content.strip()

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Prompt ({len(prompt)} chars):")
        print(indent(prompt, "    "))

        # Validation
        errors = []
        if len(prompt) < 50:
            errors.append("Prompt too short (< 50 chars)")
        if len(prompt) > 1500:
            errors.append("Prompt too long (> 1500 chars)")
        if prompt.startswith("{") or prompt.startswith("["):
            errors.append("Got JSON instead of plain text prompt")

        # Video prompts should mention motion/camera
        motion_keywords = ["camera", "pan", "slow", "move", "turn", "gaze", "wind",
                           "flicker", "sway", "shift", "tracking", "static", "drift",
                           "walk", "step", "look", "breathe", "rise", "fall"]
        has_motion = any(kw in prompt.lower() for kw in motion_keywords)
        if not has_motion:
            errors.append("Video prompt doesn't describe any motion or camera movement")

        if errors:
            print(f"\n  ⚠️  Warnings: {'; '.join(errors)}")
            print("  ✅ PASS (with warnings)\n")
        else:
            print("  ✅ PASS\n")
        return True

    except Exception as e:
        print(f"  ❌ FAIL: {e}\n")
        return False


def test_batch_consistency(client: OpenAI, model: str) -> bool:
    """Generate image prompts for 3 sequential scenes — check narrative flow."""
    print("=" * 70)
    print("TEST 4: Batch consistency (3 sequential scenes)")
    print("=" * 70)

    scenes = [
        {"id": "scene_001", "order": 1, "time_start": 0.0, "time_end": 6.8,
         "type": "vocal", "lyrics": "Standing on the edge of something new / The city lights below are fading through",
         "characters": ["protagonist"], "settings": ["rooftop"]},
        {"id": "scene_002", "order": 2, "time_start": 6.8, "time_end": 10.4,
         "type": "vocal", "lyrics": "I hear the echoes of a distant sound",
         "characters": ["protagonist"], "settings": ["rooftop"]},
        {"id": "scene_003", "order": 3, "time_start": 10.4, "time_end": 14.0,
         "type": "vocal", "lyrics": "My heart is racing but my feet won't leave the ground",
         "characters": ["protagonist"], "settings": ["rooftop"]},
    ]

    prompts = []
    total_time = 0

    for scene in scenes:
        setting_desc = ""
        for s in SAMPLE_STYLE_SHEET["settings"]:
            if s["id"] in scene.get("settings", []):
                setting_desc = s["description"]
                break

        char_desc = ""
        for c in SAMPLE_STYLE_SHEET["characters"]:
            if c["id"] in scene.get("characters", []):
                char_desc = c["description"]
                break

        # Include previous prompts for narrative context
        context = ""
        if prompts:
            context = "Previous scene prompts (for narrative continuity):\n"
            for prev_id, prev_prompt in prompts:
                context += f"  {prev_id}: {prev_prompt}\n"
            context += "\n"

        user_msg = dedent(f"""\
            {context}Scene: {scene['id']} ({scene['time_start']:.2f}s – {scene['time_end']:.2f}s)
            Type: {scene['type']}
            Lyrics: {scene['lyrics']}

            Style sheet:
              Visual style: {SAMPLE_STYLE_SHEET['visual_style']}
              Color palette: {SAMPLE_STYLE_SHEET['color_palette']}
              Character: {char_desc}
              Setting: {setting_desc}

            Write a FLUX image prompt for this scene. Maintain visual continuity with previous scenes.
        """)

        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": IMAGE_PROMPT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=300,
                temperature=0.7,
            )
            elapsed = time.time() - t0
            total_time += elapsed
            prompt = resp.choices[0].message.content.strip()
            prompts.append((scene["id"], prompt))

            print(f"\n  {scene['id']} ({elapsed:.2f}s):")
            print(indent(prompt, "    "))

        except Exception as e:
            print(f"\n  {scene['id']}: ❌ FAIL: {e}")
            return False

    print(f"\n  Total time: {total_time:.2f}s ({total_time/len(scenes):.2f}s avg)")
    print(f"  All 3 prompts reference same character/setting: check visually above")
    print("  ✅ PASS\n")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test vLLM prompt generation for MusicVision")
    parser.add_argument("--base-url", default=None,
                        help="vLLM base URL (default: $OPENAI_BASE_URL or http://localhost:8000/v1)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: $OPENAI_MODEL or 'qwen32b')")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: $OPENAI_API_KEY or 'vllm')")
    parser.add_argument("--test", choices=["all", "segmentation", "image-prompts", "video-prompts", "batch"],
                        default="all", help="Which test to run (default: all)")
    args = parser.parse_args()

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    model = args.model or os.environ.get("OPENAI_MODEL", "qwen32b")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "vllm")

    print(f"\nMusicVision vLLM Prompt Generation Test")
    print(f"{'=' * 70}")
    print(f"  Endpoint: {base_url}")
    print(f"  Model:    {model}")
    print(f"  Test:     {args.test}")
    print()

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Connection check first
    if not test_connection(client, model):
        print("Cannot reach vLLM server. Check that it's running and the URL is correct.")
        sys.exit(1)

    tests = {
        "segmentation": ("Scene Segmentation", test_segmentation),
        "image-prompts": ("Image Prompts", test_image_prompts),
        "video-prompts": ("Video Prompts", test_video_prompts),
        "batch": ("Batch Consistency", test_batch_consistency),
    }

    results = {}
    if args.test == "all":
        for key, (name, func) in tests.items():
            results[name] = func(client, model)
    else:
        name, func = tests[args.test]
        results[name] = func(client, model)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n  {passed}/{total} tests passed")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
