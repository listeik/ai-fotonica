import csv
import json

import pytest
from doe.benchmark import main


def test_classical_benchmark_exports_case_metrics_and_provenance(tmp_path):
    pytest.importorskip("matplotlib")
    main(
        [
            "--split",
            "val",
            "--samples",
            "1",
            "--classical-steps",
            "1",
            "--output",
            str(tmp_path),
        ]
    )
    report = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    with (tmp_path / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        (csv_case,) = csv.DictReader(stream)
    (case,) = report["samples"]
    (summary,) = report["summary"]
    assert report["metadata"]["split"] == "val"
    assert report["metadata"]["checkpoint"] is None
    assert case["iterations"] == 1
    assert len(case["target_sha256"]) == len(case["phase_png_sha256"]) == 64
    assert len(case["planes"]) == 4
    assert summary["worst_pcc"] == min(plane["pcc"] for plane in case["planes"])
    assert float(csv_case["plane_4_ssim"]) == case["planes"][3]["ssim"]
    assert (tmp_path / "comparison.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "Classical (1)" in (tmp_path / "report.md").read_text(encoding="utf-8")
