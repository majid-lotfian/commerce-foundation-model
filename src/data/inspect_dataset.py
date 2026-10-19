
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm

def read_jsonl_gz(path):
    """Stream compressed JSONL records with a progress bar."""
    path = Path(path)

    with open(path, "rb") as raw:
        with tqdm(
            total=path.stat().st_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=path.name,
        ) as progress:
            with gzip.GzipFile(fileobj=raw, mode="rb") as f:
                previous_position = 0

                for line in f:
                    if line.strip():
                        yield json.loads(line)

                    current_position = raw.tell()
                    progress.update(
                        current_position - previous_position
                    )
                    previous_position = current_position


def inspect_dataset(category, data_dir):
    data_dir = Path(data_dir)

    reviews_path = data_dir / "reviews" / f"{category}.jsonl.gz"
    metadata_path = data_dir / "metadata" / f"meta_{category}.jsonl.gz"

    if not reviews_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing reviews or metadata for category: {category}"
        )

    # Step 1: Inspect product metadata
    product_ids = set()
    total_products = 0
    products_with_price = 0
    products_with_description = 0

    for product in read_jsonl_gz(metadata_path):
        total_products += 1

        if product.get("parent_asin"):
            product_ids.add(product["parent_asin"])

        if product.get("price") is not None:
            products_with_price += 1

        if product.get("description"):
            products_with_description += 1

    # Step 2: Inspect reviews and user interactions
    total_reviews = 0
    verified_purchases = 0
    matched_reviews = 0
    user_review_counts = Counter()
    ratings = Counter()

    for review in read_jsonl_gz(reviews_path):
        total_reviews += 1

        user_id = review.get("user_id")
        if user_id:
            user_review_counts[user_id] += 1

        if review.get("verified_purchase") is True:
            verified_purchases += 1

        if review.get("parent_asin") in product_ids:
            matched_reviews += 1

        if review.get("rating") is not None:
            ratings[review["rating"]] += 1

    # Step 3: Calculate statistics
    unique_users = len(user_review_counts)

    users_2_plus = sum(
        count >= 2 for count in user_review_counts.values()
    )
    users_5_plus = sum(
        count >= 5 for count in user_review_counts.values()
    )
    users_10_plus = sum(
        count >= 10 for count in user_review_counts.values()
    )

    def percentage(value, total):
        return 100 * value / total if total else 0

    print(f"\n{'=' * 55}")
    print(f"Amazon Reviews 2023 — {category}")
    print(f"{'=' * 55}")

    print(f"Total reviews:             {total_reviews:,}")
    print(f"Total metadata records:    {total_products:,}")
    print(f"Unique product IDs:        {len(product_ids):,}")
    print(f"Unique users:              {unique_users:,}")

    print("\nInteraction statistics")
    print(f"Verified purchases:        {percentage(verified_purchases, total_reviews):.2f}%")
    print(f"Reviews linked to product: {percentage(matched_reviews, total_reviews):.2f}%")
    print(f"Avg reviews per user:      {total_reviews / unique_users if unique_users else 0:.2f}")

    print("\nUser history lengths")
    print(f"Users with 2+ reviews:     {users_2_plus:,} ({percentage(users_2_plus, unique_users):.2f}%)")
    print(f"Users with 5+ reviews:     {users_5_plus:,} ({percentage(users_5_plus, unique_users):.2f}%)")
    print(f"Users with 10+ reviews:    {users_10_plus:,} ({percentage(users_10_plus, unique_users):.2f}%)")

    print("\nProduct metadata completeness")
    print(f"Products with price:       {percentage(products_with_price, total_products):.2f}%")
    print(f"Products with description: {percentage(products_with_description, total_products):.2f}%")

    print("\nRating distribution")
    for rating, count in sorted(ratings.items()):
        print(f"  {rating}: {count:,} ({percentage(count, total_reviews):.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--data-dir", default="data/raw")

    args = parser.parse_args()

    if "all" in args.categories:
        categories = sorted(
            p.name.removesuffix(".jsonl.gz")
            for p in (Path(args.data_dir) / "reviews").glob("*.jsonl.gz")
        )
    else:
        categories = args.categories

    for category in categories:
        inspect_dataset(category, args.data_dir)
