"""Bias-signal detector tests."""

from backend.agents.bias_agent import detect_bias_signals


def test_empty_input_returns_no_flags() -> None:
    assert detect_bias_signals("") == []


def test_clean_resume_no_flags() -> None:
    """A neutral resume must not produce any spurious flags."""
    text = (
        "John Doe\nSoftware Engineer\nBuilt distributed systems at scale.\n"
        "Bachelor of Science, Computer Science."
    )
    assert detect_bias_signals(text) == []


def test_dob_flagged() -> None:
    flags = detect_bias_signals("DOB: 1990-05-12")
    assert any("birth" in f.lower() for f in flags)


def test_explicit_gender_flagged() -> None:
    flags = detect_bias_signals("Gender: female")
    assert any("gender" in f.lower() for f in flags)


def test_grad_year_flagged() -> None:
    flags = detect_bias_signals("Graduated: 2012")
    assert any("graduation" in f.lower() for f in flags)


def test_marital_status_flagged() -> None:
    flags = detect_bias_signals("Marital status: married")
    assert any("marital" in f.lower() for f in flags)


def test_full_address_flagged() -> None:
    flags = detect_bias_signals("Address: 123 Main Street")
    assert any("address" in f.lower() for f in flags)


def test_photo_reference_flagged() -> None:
    flags = detect_bias_signals("Photograph attached on the back page.")
    assert any("photo" in f.lower() for f in flags)


def test_balanced_gendered_language_not_flagged() -> None:
    """Balanced gendered language (no 2× dominance) must NOT be flagged."""
    text = (
        "A driven, ambitious leader who is also collaborative, supportive, "
        "and empathetic. Confident yet kind, decisive yet considerate."
    )
    flags = detect_bias_signals(text)
    assert not any("masculine-coded" in f or "feminine-coded" in f for f in flags)


def test_heavily_masculine_coded_flagged() -> None:
    """Strong dominance of masculine-coded terms should trigger the flag."""
    text = (
        "Aggressive, ambitious, dominant, fearless, decisive, headstrong leader "
        "with autonomous and competitive instincts."
    )
    flags = detect_bias_signals(text)
    assert any("masculine-coded" in f for f in flags)


def test_combined_signals_produces_multiple_flags() -> None:
    """Realistic over-disclosure resume should produce ≥3 distinct flags."""
    text = (
        "Jane Doe\nDOB: 1990-05-12\nGender: female\nMarital status: married\n"
        "Address: 123 Main Street\nClass of 2012"
    )
    flags = detect_bias_signals(text)
    assert len(flags) >= 4
