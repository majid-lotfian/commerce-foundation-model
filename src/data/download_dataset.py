import argparse
import time
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"

CATEGORIES = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
]


def download_file(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    if destination.exists():
        print(f"Already downloaded: {destination}")
        return

    for attempt in range(1, 4):
        try:
            existing = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}

            with requests.get(
                url, headers=headers, stream=True, timeout=(30, 120)
            ) as response:
                response.raise_for_status()

                # Append only when the server honors the byte range.
                append = existing > 0 and response.status_code == 206
                mode = "ab" if append else "wb"

                content_length = int(response.headers.get("Content-Length", 0))
                total_size = (
                    existing + content_length if append else content_length
                )

                with tqdm(
                    total=total_size or None,
                    initial=existing if append else 0,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=destination.name,
                ) as progress:
                    with open(partial, mode) as file:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                file.write(chunk)
                                progress.update(len(chunk))

            partial.rename(destination)
            print(f"Completed: {destination}")

            return

        except (requests.RequestException, OSError) as error:
            print(f"Attempt {attempt} failed: {error}")
            if attempt < 3:
                time.sleep(5 * attempt)

    raise RuntimeError(f"Download failed: {url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--categories", nargs="+", default=["All_Beauty"])
    args = parser.parse_args()

    output = Path(args.output)

    categories = CATEGORIES if "all" in args.categories else args.categories

    for category in categories:
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category: {category}")

        files = [
            (
                f"{BASE_URL}/review_categories/{category}.jsonl.gz",
                output / "reviews" / f"{category}.jsonl.gz",
            ),
            (
                f"{BASE_URL}/meta_categories/meta_{category}.jsonl.gz",
                output / "metadata" / f"meta_{category}.jsonl.gz",
            ),
        ]

        for url, destination in files:
            download_file(url, destination)


if __name__ == "__main__":
    main()
