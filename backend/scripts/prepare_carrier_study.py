"""Prepare source-separated carrier-study tensors without scoring held-out targets.

Provide explicit paths to the original runs directory, verified photo-quality-v2
data and the new output directory. BSDS500 images are fetched from the Berkeley
BIDS institutional mirror at a pinned Git commit. Its train, validation and test
images remain separate. Every downloaded file and prepared target receives a
SHA-256 record in the output manifest.
"""

import argparse
import csv
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import torch
from PIL import Image

SOURCE_PAGE = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html"
BSDS_COMMIT = "a04b7c6c3a9f0ace74bf205c72a43d32e1c72722"
EXTERNAL_INDICES = (0, 44, 88, 131)
EXTERNAL_NAMES = ("100007.jpg", "146074.jpg", "207038.jpg", "290035.jpg")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save(path, data):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)
    assert torch.load(path, map_location="cpu", weights_only=True).keys() == data.keys()


def pixels(data):
    with Image.open(BytesIO(data)) as image:
        assert image.size == (128, 128) and image.mode == "L"
        return torch.from_numpy(np.array(image, copy=True))


def select_unique(rows, split, excluded_indices, excluded_origins, count, seed):
    candidates = []
    for (part, index), planes in rows.items():
        if part != split or index in excluded_indices:
            continue
        origins = {(r["source"], r["origin"]) for r in planes.values()}
        if len(origins) != 4 or origins & excluded_origins:
            continue
        candidates.append(index)
    rng = np.random.default_rng(seed)
    selected, used = [], set()
    for index in rng.permutation(sorted(candidates)).tolist():
        origins = {(r["source"], r["origin"]) for r in rows[(split, index)].values()}
        if origins & used:
            continue
        selected.append(index)
        used.update(origins)
        if len(selected) == count:
            break
    assert len(selected) == count, (split, len(selected), len(candidates))
    return sorted(selected), len(candidates), used


def load_selected(root, rows, split, selected, forbidden_pixels):
    stacks = []
    for index in selected:
        stack = []
        for plane in range(4):
            row = rows[(split, index)][plane]
            path = root / split / f"{index // 100:03d}" / f"{index:05d}_{plane + 1}.png"
            image = pixels(path.read_bytes())
            checksum = hashlib.sha256(image.numpy().tobytes()).hexdigest()
            assert checksum == row["sha256"]
            assert checksum not in forbidden_pixels
            stack.append(image)
        stacks.append(torch.stack(stack))
    return torch.stack(stacks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="Original runs directory containing photo-model-v1 and DOE source datasets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for frozen tensors, provenance and downloaded sources",
    )
    parser.add_argument(
        "--previous",
        type=Path,
        required=True,
        help="Verified photo-quality-v2 data directory with train-val.pt, test.pt and manifest.json",
    )
    args = parser.parse_args()
    runs, output, previous = args.runs.resolve(), args.output.resolve(), args.previous.resolve()
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    old = read_json(previous / "manifest.json")
    assert old["heldout_ready"]
    assert digest(previous / "train-val.pt") == old["train_val_sha256"]
    assert digest(previous / "test.pt") == old["test_sha256"]
    train = torch.load(previous / "train-val.pt", map_location="cpu", weights_only=True)
    tests = torch.load(previous / "test.pt", map_location="cpu", weights_only=True)
    base = read_json(runs / "photo-model-v1/data.json")
    broad_root = runs / "doe-dataset-v1"
    manifest_path = broad_root / "manifest.csv.gz"
    assert digest(manifest_path) == base["broad_manifest_sha256"]
    assert digest(runs / "people-doe-100-v1/manifest.json") == base["people_manifest_sha256"]
    assert digest(runs / "mixed-doe-50-v1.zip") == old["source_sha256"]["mixed_archive"]
    cache_path = runs / "photo-model-v1/cache/broad-train.npy"
    broad_train = np.load(cache_path, mmap_mode="r")
    selected_train = {
        int(index): position for position, index in enumerate(base["broad_indices"]["train"])
    }
    natural_images, natural_records, natural_hashes = [], [], set()
    assert digest(cache_path) == old["broad_cache_sha256"]["train"]
    rows, origin_splits, all_pixels, training_pixels = {}, {}, set(), set()
    with gzip.open(manifest_path, "rt", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            split, index, plane = row["split"], int(row["set"]), int(row["plane"]) - 1
            origin = (row["source"], row["origin"])
            assert origin_splits.setdefault(origin, split) == split
            rows.setdefault((split, index), {})[plane] = row
            all_pixels.add(row["sha256"])
            if split == "train":
                training_pixels.add(row["sha256"])
                if row["source"] in ("open_images", "usda_ars") and index in selected_train:
                    image = np.array(broad_train[selected_train[index], plane], copy=True)
                    checksum = hashlib.sha256(image.tobytes()).hexdigest()
                    assert checksum == row["sha256"]
                    if checksum not in natural_hashes:
                        natural_hashes.add(checksum)
                        natural_images.append(torch.from_numpy(image))
                        natural_records.append(
                            {
                                "source": row["source"],
                                "origin": row["origin"],
                                "set": index,
                                "plane": plane + 1,
                                "pixel_sha256": checksum,
                            }
                        )

    excluded = {
        "validation": set(base["broad_indices"]["validation"]),
        "test": set(base["broad_indices"]["test"]) | set(old["fresh_broad_indices"]),
    }
    excluded["test"].update(old["prior_fourier_broad_test_indices"])
    prior_manifests = {}
    for path in sorted((runs / "fourier-quality-v1").glob("*/data.json")):
        candidate = read_json(path)
        if "splits" not in candidate:
            continue
        assert candidate["broad_manifest_sha256"] == base["broad_manifest_sha256"]
        prior_manifests[str(path)] = digest(path)
        for split in excluded:
            for entry in candidate["splits"].get(split, []):
                if entry["source"] == "broad":
                    index = int(entry["index"])
                    assert [rows[(split, index)][i]["sha256"] for i in range(4)] == entry[
                        "pixel_sha256"
                    ]
                    excluded[split].add(index)
    excluded_origins = {
        split: {
            (r["source"], r["origin"]) for index in selected for r in rows[(split, index)].values()
        }
        for split, selected in excluded.items()
    }
    from zipfile import ZipFile

    with ZipFile(runs / "mixed-doe-50-v1.zip") as archive:
        mixed = json.loads(archive.read("mixed-doe-50-v1/manifest.json"))
        for record in mixed["records"]:
            split = record["split"]
            if split not in excluded:
                continue
            for target in record["targets"]:
                source = target["source"]
                parent = source.get("parent", "")
                if parent and source["dataset"] != "Olivetti":
                    index = int(Path(parent).stem.split("_")[0])
                    excluded[split].add(index)
                    excluded_origins[split].update(
                        (r["source"], r["origin"]) for r in rows[(split, index)].values()
                    )
    val_ids, val_candidates, val_sources = select_unique(
        rows, "validation", excluded["validation"], excluded_origins["validation"], 256, 20261117
    )
    test_ids, test_candidates, test_sources = select_unique(
        rows, "test", excluded["test"], excluded_origins["test"], 256, 20261118
    )
    train["validation"]["new_broad_val"] = load_selected(
        broad_root, rows, "validation", val_ids, training_pixels
    )
    fresh_test = load_selected(broad_root, rows, "test", test_ids, training_pixels)
    assert len(val_sources) == len(test_sources) == 1024
    assert not val_sources & test_sources
    assert not val_sources & excluded_origins["validation"]
    assert not test_sources & excluded_origins["test"]

    source_dir = output / "sources"
    source_dir.mkdir(exist_ok=True)
    download_path = source_dir / "bsds-download.json"
    if not download_path.exists():
        with httpx.Client(trust_env=False, timeout=60) as client:
            response = client.get(
                f"https://api.github.com/repos/BIDS/BSDS500/git/trees/{BSDS_COMMIT}?recursive=1"
            )
            response.raise_for_status()
            paths = [
                item["path"]
                for item in response.json()["tree"]
                if item["path"].startswith("BSDS500/data/images/") and item["path"].endswith(".jpg")
            ]
        wanted = []
        for split, count in (("train", 200), ("val", 32), ("test", 132)):
            wanted.extend(sorted(path for path in paths if f"/{split}/" in path)[:count])
        assert len(wanted) == 364
        write_json(
            download_path,
            {
                "repository": "https://github.com/BIDS/BSDS500",
                "commit": BSDS_COMMIT,
                "downloaded_paths": wanted,
            },
        )
    download_manifest = read_json(download_path)

    def download_image(name):
        path = source_dir / "bsds-originals" / Path(name).relative_to("BSDS500/data/images")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/BIDS/BSDS500/{BSDS_COMMIT}/{name}"
            with httpx.Client(trust_env=False, timeout=60) as client:
                response = client.get(url)
                response.raise_for_status()
                temporary = path.with_suffix(".download")
                temporary.write_bytes(response.content)
                temporary.replace(path)

    with ThreadPoolExecutor(8) as executor:
        list(executor.map(download_image, download_manifest["downloaded_paths"]))
    assert download_manifest["commit"] == BSDS_COMMIT
    bsds_root = source_dir / "bsds-originals"
    paths_by_split = {}
    for split, count in (("train", 200), ("val", 32), ("test", 132)):
        paths_by_split[split] = sorted((bsds_root / split).glob("*.jpg"))
        assert len(paths_by_split[split]) == count
    external_records, bsds_pixels, bsds_original_hashes, bsds_source_pixels = [], {}, set(), set()
    for split, paths in paths_by_split.items():
        images = []
        for source_file in paths:
            data = source_file.read_bytes()
            file_hash = hashlib.sha256(data).hexdigest()
            assert file_hash not in bsds_original_hashes
            bsds_original_hashes.add(file_hash)
            with Image.open(BytesIO(data)) as original:
                rgb_hash = hashlib.sha256(original.convert("RGB").tobytes()).hexdigest()
                assert rgb_hash not in bsds_source_pixels
                bsds_source_pixels.add(rgb_hash)
                image = original.convert("L").resize((128, 128), Image.Resampling.LANCZOS)
            target = torch.from_numpy(np.array(image, copy=True))
            checksum = hashlib.sha256(target.numpy().tobytes()).hexdigest()
            assert checksum not in all_pixels and checksum not in natural_hashes
            natural_hashes.add(checksum)
            target_file = source_dir / "bsds-gray128" / split / (source_file.stem + ".png")
            target_file.parent.mkdir(parents=True, exist_ok=True)
            image.save(target_file)
            images.append(target)
            external_records.append(
                {
                    "dataset": "BSDS500 Berkeley institutional BIDS mirror",
                    "split": split,
                    "source_file": source_file.name,
                    "source_url": f"https://raw.githubusercontent.com/BIDS/BSDS500/{BSDS_COMMIT}/BSDS500/data/images/{split}/{source_file.name}",
                    "official_source_page": SOURCE_PAGE,
                    "source_file_sha256": file_hash,
                    "source_rgb_pixel_sha256": rgb_hash,
                    "target_png_file": str(target_file.relative_to(output)),
                    "target_png_sha256": digest(target_file),
                    "target_pixel_sha256": checksum,
                    "processing": "Pillow convert L then Lanczos resize128x128; uint8 fixed scale; no contrast normalization",
                }
            )
            if split == "train":
                natural_images.append(target)
                natural_records.append(
                    {
                        "source": "bsds500_train",
                        "origin": source_file.name,
                        "pixel_sha256": checksum,
                    }
                )
        bsds_pixels[split] = torch.stack(images)
    train["natural_bank"] = torch.stack(natural_images)
    train["validation"]["natural_val"] = bsds_pixels["val"].reshape(8, 4, 128, 128)
    assert tuple(paths_by_split["test"][index].name for index in EXTERNAL_INDICES) == EXTERNAL_NAMES
    remaining_indices = [index for index in range(132) if index not in EXTERNAL_INDICES]
    external = bsds_pixels["test"][list(EXTERNAL_INDICES)]
    tests["test"]["previous_fresh_broad_test"] = tests["test"].pop("fresh_broad_test")
    tests["test_ids"]["previous_fresh_broad_test"] = tests["test_ids"].pop("fresh_broad_test")
    tests["test"]["fresh_broad_test"] = fresh_test
    tests["test_ids"]["fresh_broad_test"] = [f"test/{i:05d}" for i in test_ids]
    tests["test"]["external_quartet"] = external[None]
    tests["test"]["external_natural_test"] = bsds_pixels["test"][remaining_indices].reshape(
        32, 4, 128, 128
    )
    tests["test_ids"]["external_natural_test"] = [
        "BSDS500/"
        + "+".join(paths_by_split["test"][index].name for index in remaining_indices[i : i + 4])
        for i in range(0, 128, 4)
    ]
    tests["test_ids"]["external_quartet"] = ["BSDS500/" + "+".join(EXTERNAL_NAMES)]
    train_path = output / "train-val.pt"
    if train_path.exists():
        previous_train = torch.load(train_path, map_location="cpu", weights_only=True)
        for key, tensor in train.items():
            if isinstance(tensor, dict):
                assert tensor.keys() == previous_train[key].keys()
                assert all(
                    torch.equal(value, previous_train[key][name]) for name, value in tensor.items()
                )
            else:
                assert torch.equal(tensor, previous_train[key]), "Prepared training data changed"
    else:
        save(train_path, train)
    save(output / "test.pt", tests)
    report = {
        "version": "carrier-study-v1-data-v1",
        "broad_train_npy": str(cache_path),
        "broad_train_sha256": digest(cache_path),
        "broad_cache_sha256": {"train": digest(cache_path)},
        "broad_train_count": len(np.load(cache_path, mmap_mode="r")),
        "previous_prepared_data": str(previous),
        "previous_manifest_sha256": digest(previous / "manifest.json"),
        "previous_train_val_sha256": old["train_val_sha256"],
        "previous_test_sha256": old["test_sha256"],
        "source_sha256": old["source_sha256"],
        "prior_fourier_manifests": prior_manifests,
        "source_broad_manifest_sha256": digest(manifest_path),
        "training_counts": old["training_counts"],
        "validation_counts": {k: len(v) for k, v in train["validation"].items()},
        "test_counts": {k: len(v) for k, v in tests["test"].items()},
        "new_validation": {
            "seed": 20261117,
            "indices": val_ids,
            "eligible_quartets": val_candidates,
            "unique_source_origins": len(val_sources),
            "prior_excluded_indices": sorted(excluded["validation"]),
        },
        "fresh_test": {
            "seed": 20261118,
            "indices": test_ids,
            "eligible_quartets": test_candidates,
            "unique_source_origins": len(test_sources),
            "prior_excluded_indices": sorted(excluded["test"]),
        },
        "natural_bank": {
            "count": len(natural_images),
            "records": natural_records,
            "original_photo_origin_count": 248,
            "fashion_images_in_natural_bank": 0,
            "bsds_train_origin_count": 200,
        },
        "external": {
            "repository_commit": BSDS_COMMIT,
            "download_manifest_sha256": digest(source_dir / "bsds-download.json"),
            "records": external_records,
            "selection": "Sorted first32 validation files form8quartets; sorted test indices0,44,88,131 form requested quartet; remaining128test files form32additional quartets. IDs frozen before any model inference.",
            "benchmark_notice": "BSDS500 is a public benchmark; source has no declared overlap with project training origins.",
        },
        "portrait_subjects_by_split": old["portrait_subjects_by_split"],
        "train_val_sha256": digest(output / "train-val.pt"),
        "test_sha256": digest(output / "test.pt"),
        "heldout_ready": True,
        "caveats": [
            "Distinct source IDs and pixel hashes do not prove semantic independence of related photographs.",
            "The fixed external quartet includes public benchmark photographs of people, used without identity inference.",
            "Original diagnostic heldouts were previously scored and must not select models.",
            "Fresh broad test and new broad validation are new independent-source quartets within the original broad distribution.",
            "External quartet contains four fixed images; additional32quartets provide a broader external natural-image check.",
        ],
    }
    write_json(output / "manifest.json", report)
    print(
        json.dumps(
            {
                "output": str(output),
                "validation": report["validation_counts"],
                "test": report["test_counts"],
                "train_val_sha256": report["train_val_sha256"],
                "test_sha256": report["test_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
