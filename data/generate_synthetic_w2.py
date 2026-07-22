"""
Synthetic W-2 document generator.

Why synthetic: real annotated tax/insurance forms are hard to source publicly
(sensitive data). This generator produces an unlimited number of W-2-style
documents as PDFs, each with a paired ground-truth JSON, so the eval harness
has perfect labels to score against.

Layout is loosely modeled on the real IRS Form W-2 field set (box 1 wages,
box 2 federal tax withheld, EIN, SSN-shaped identifier) but uses fully fake
data — no real PII, no IRS branding, safe to publish.

Usage:
    python data/generate_synthetic_w2.py --n 50 --out data/synthetic_w2_train
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker()


@dataclass
class SyntheticW2:
    document_type: str
    payer_name: str
    recipient_name: str
    identifier: str  # fake SSN-shaped id, NOT a real SSN
    tax_year: str
    total_amount: str
    federal_tax_withheld: str
    employer_ein: str


def _fake_ssn_shaped_id() -> str:
    return f"{random.randint(100,899):03d}-{random.randint(10,99):02d}-{random.randint(1000,9999):04d}"


def _fake_ein() -> str:
    return f"{random.randint(10,99):02d}-{random.randint(1000000,9999999):07d}"


def generate_record(tax_year: int) -> SyntheticW2:
    wages = round(random.uniform(28000, 180000), 2)
    withheld = round(wages * random.uniform(0.10, 0.24), 2)
    return SyntheticW2(
        document_type="W-2",
        payer_name=fake.company(),
        recipient_name=fake.name(),
        identifier=_fake_ssn_shaped_id(),
        tax_year=str(tax_year),
        total_amount=f"{wages:,.2f}",
        federal_tax_withheld=f"{withheld:,.2f}",
        employer_ein=_fake_ein(),
    )


def render_pdf(record: SyntheticW2, path: Path, noise_level: float = 0.0) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    def jitter(v: float) -> float:
        return v + random.uniform(-noise_level, noise_level) if noise_level else v

    c.setFont("Helvetica-Bold", 14)
    c.drawString(jitter(72), height - 72, "Form W-2  Wage and Tax Statement (SYNTHETIC - sample data only)")

    c.setFont("Helvetica", 10)
    y = height - 110
    line_height = 20
    rows = [
        ("a Employee's SSN", record.identifier),
        ("b Employer EIN", record.employer_ein),
        ("c Employer name", record.payer_name),
        ("e Employee name", record.recipient_name),
        ("Tax year", record.tax_year),
        ("1 Wages, tips, other compensation", record.total_amount),
        ("2 Federal income tax withheld", record.federal_tax_withheld),
    ]
    for label, value in rows:
        c.drawString(jitter(72), jitter(y), f"{label}:")
        c.drawString(jitter(320), jitter(y), str(value))
        y -= line_height

    c.showPage()
    c.save()


def generate_dataset(n: int, out_dir: Path, noise_level: float = 0.0, seed: int = 0) -> None:
    random.seed(seed)
    Faker.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "documents").mkdir(exist_ok=True)
    (out_dir / "ground_truth").mkdir(exist_ok=True)

    manifest = []
    for i in range(n):
        tax_year = random.choice([2023, 2024, 2025])
        record = generate_record(tax_year)
        doc_id = f"w2_{i:04d}"
        pdf_path = out_dir / "documents" / f"{doc_id}.pdf"
        gt_path = out_dir / "ground_truth" / f"{doc_id}.json"

        render_pdf(record, pdf_path, noise_level=noise_level)
        gt_path.write_text(json.dumps(asdict(record), indent=2))
        manifest.append({"doc_id": doc_id, "pdf": str(pdf_path), "ground_truth": str(gt_path)})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Generated {n} synthetic W-2 documents in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", type=str, default="data/synthetic_w2")
    parser.add_argument("--noise", type=float, default=0.0, help="layout jitter for a noisier eval split")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate_dataset(n=args.n, out_dir=Path(args.out), noise_level=args.noise, seed=args.seed)
