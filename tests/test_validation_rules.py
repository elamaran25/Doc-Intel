from extraction.schemas import ExtractedDocument, ExtractedField, FieldConfidence
from validation.agent import _validate_identifier_format, _validate_required_fields


def _make_doc(**overrides) -> ExtractedDocument:
    base = dict(
        document_type=ExtractedField(value="W-2"),
        payer_name=ExtractedField(value="Acme Corp"),
        recipient_name=ExtractedField(value="Jane Doe"),
        identifier=ExtractedField(value="123-45-6789"),
        tax_year=ExtractedField(value="2025"),
        total_amount=ExtractedField(value="54000.00"),
    )
    base.update(overrides)
    return ExtractedDocument(**base)


def test_valid_identifier_passes():
    doc = _make_doc()
    assert _validate_identifier_format(doc) is None


def test_malformed_identifier_flagged():
    doc = _make_doc(identifier=ExtractedField(value="not-a-valid-id!!"))
    issue = _validate_identifier_format(doc)
    assert issue is not None
    assert issue.field_name == "identifier"


def test_missing_required_field_flagged():
    doc = _make_doc(payer_name=ExtractedField(value=None, confidence=FieldConfidence.LOW))
    issues = _validate_required_fields(doc)
    assert any(i.field_name == "payer_name" for i in issues)
