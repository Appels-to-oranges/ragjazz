"""
Download ~20 Wikipedia articles on jazz piano and piano music theory.
Each article is saved as a JSON file with full text, section breakdowns, and metadata.
"""

import json
import os
import time
from datetime import datetime, timezone

import wikipediaapi

# Curated jazz-piano theory set: core harmony, scales, piano technique,
# and a few advanced topics — comprehensive without adjacent fluff.
ARTICLES = [
    # Core / existing
    "Jazz piano",
    "Jazz harmony",
    "Chord (music)",
    "Chord progression",
    "Ii–V–I progression",
    "Circle of fifths",
    "Blues scale",
    "Bebop scale",
    "Jazz improvisation",
    "Voicing (music)",
    "Comping (jazz)",
    "Stride piano",
    "Boogie-woogie",
    "Modal jazz",
    "Twelve-bar blues",
    "Tritone substitution",
    "Jazz scale",
    "Pentatonic scale",
    "Rhythm changes",
    # Note: "Piano blues" redirects to the giant general Blues article — skip it;
    # twelve-bar blues + blues scale already cover that ground.
    # Expansion: harmony building blocks
    "Seventh chord",
    "Extended chord",
    "Secondary dominant",
    "Altered chord",
    "Altered scale",
    "Voice leading",
    # Expansion: jazz-specific progressions & concepts
    "Turnaround (music)",
    "Coltrane changes",
    "Backdoor progression",
    "Chord-scale system",
    # Expansion: piano voicing / texture
    "Upper structure",
    "Quartal harmony",
    "Block chord",
    "Walking bass",
    # Expansion: beginner foundations + slash chords
    # Skip: Reharmonization→Harmonization, Guide tone→Jazz improvisation,
    # Drop 2 / Shell chord (no usable Wikipedia pages).
    "Interval (music)",
    "Triad (music)",
    "Key signature",
    "Major scale",
    "Minor scale",
    "Slash chord",
]

OUTPUT_DIR = "articles"


def extract_sections(section, depth=0):
    """Recursively extract section title, text, and subsections."""
    return {
        "title": section.title,
        "depth": depth,
        "text": section.text,
        "subsections": [
            extract_sections(s, depth + 1) for s in section.sections
        ],
    }


def safe_get_categories(page):
    try:
        return list(page.categories.keys())
    except Exception:
        return []


def safe_get_links_count(page):
    try:
        return len(page.links)
    except Exception:
        return 0


def download_article(wiki, title, retries=3):
    for attempt in range(retries):
        try:
            page = wiki.page(title)
            if not page.exists():
                print(f"  [SKIP] '{title}' not found")
                return None

            sections = [extract_sections(s) for s in page.sections]

            article = {
                "metadata": {
                    "title": page.title,
                    "url": page.fullurl,
                    "summary": page.summary,
                    "categories": safe_get_categories(page),
                    "links_count": safe_get_links_count(page),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "source": "wikipedia",
                    "language": "en",
                },
                "full_text": page.text,
                "sections": sections,
            }
            return article
        except Exception as e:
            print(f"  [RETRY {attempt + 1}/{retries}] {e}")
            time.sleep(2 * (attempt + 1))

    print(f"  [FAIL] Could not download '{title}' after {retries} attempts")
    return None


def slugify(title):
    return (
        title.lower()
        .replace(" ", "_")
        .replace("–", "-")
        .replace("(", "")
        .replace(")", "")
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wiki = wikipediaapi.Wikipedia(
        user_agent="ragjazz/1.0 (educational project; https://ragjazz.fyi ; github.com/Appels-to-oranges/ragjazz)",
        language="en",
    )

    downloaded = []
    for title in ARTICLES:
        print(f"Downloading: {title}")
        article = download_article(wiki, title)
        if article is None:
            continue

        filename = slugify(title) + ".json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(article, f, indent=2, ensure_ascii=False)

        downloaded.append(
            {
                "title": article["metadata"]["title"],
                "file": filename,
                "url": article["metadata"]["url"],
                "summary_preview": article["metadata"]["summary"][:120] + "...",
            }
        )
        print(f"  -> saved {filepath}")
        time.sleep(1)

    manifest_path = os.path.join(OUTPUT_DIR, "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_articles": len(downloaded),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "articles": downloaded,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nDone! {len(downloaded)} articles saved to '{OUTPUT_DIR}/'")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
