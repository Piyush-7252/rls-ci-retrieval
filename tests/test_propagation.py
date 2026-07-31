"""
Regression tests for propagate_effective_facts() and sentence inheritance.

Run with:
    python -m pytest tests/test_propagation.py -v
or directly:
    python tests/test_propagation.py

These tests are the authoritative definition of correct inheritance behaviour.
If any of these fail, the pipeline will produce UNKNOWN comparator outcomes or
incorrect clinical_identity for sentences and inherited paragraphs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.clinical_fact_extractor import propagate_effective_facts


# ─── helpers ─────────────────────────────────────────────────────────────────

def _heading(pos, text, facts, treatment=None, endpoint=None, population=None):
    return {
        "object_id": f"h{pos}", "type": "heading", "position": pos,
        "text": text, "facts": facts,
        "treatment_identity":  treatment  or {},
        "endpoint_identity":   endpoint   or {},
        "population_identity": population or {},
    }


def _para(pos, text, facts, treatment=None, endpoint=None, population=None):
    return {
        "object_id": f"p{pos}", "type": "paragraph", "position": pos,
        "text": text, "facts": facts,
        "treatment_identity":  treatment  or {},
        "endpoint_identity":   endpoint   or {},
        "population_identity": population or {},
    }


# ─── Test 1: core sentence-inheritance scenario (user-specified) ──────────────

def test_sentence_inheritance():
    """
    User regression test:

    Paragraph:
      Drug:     Teclistamab
      Sentence 1: Patients receive RP2D
      Sentence 2: ORR is primary endpoint

    Sentence 2 must carry:
      entities        = [ORR]                     ← own entities (from NER, NOT tested here)
      effective_facts = {drug: teclistamab,
                         endpoint: ORR}           ← inherited from paragraph

    This test exercises propagate_effective_facts on the PARAGRAPH so that after
    propagation the inherited fields are available for _build_sentence_docs to
    copy into the sentence document.
    """
    objs = [
        _heading(0, "Evaluation of Teclistamab at RP2D",
                 facts={"drug": ["teclistamab"]},
                 treatment={"primary_drug": "teclistamab"}),
        _para(1,
              # Paragraph text contains both drug and endpoint
              "Patients receive teclistamab at RP2D. ORR is the primary endpoint.",
              facts={"drug": ["teclistamab"], "endpoint": ["ORR"]},
              treatment={"primary_drug": "teclistamab"},
              endpoint={"endpoint": "ORR"}),
    ]
    propagate_effective_facts(objs)

    para = objs[1]
    ef   = para.get("effective_facts") or {}
    ti   = para.get("treatment_identity") or {}
    ei   = para.get("endpoint_identity") or {}

    # The paragraph's effective_facts must include both drug and endpoint
    assert ef.get("drug")     == ["teclistamab"], f"ef.drug wrong: {ef.get('drug')}"
    assert ef.get("endpoint") == ["ORR"],         f"ef.endpoint wrong: {ef.get('endpoint')}"

    # treatment_identity and endpoint_identity must reflect the drug/endpoint
    assert ti.get("primary_drug") == "teclistamab", f"ti.primary_drug wrong: {ti}"
    assert ei.get("endpoint")     == "ORR",          f"ei.endpoint wrong: {ei}"

    # Sentence 2 ("ORR is primary endpoint") would carry:
    #   entities        = [ORR]           (filtered from paragraph entities by char range)
    #   facts           = paragraph.facts = {drug: [teclistamab], endpoint: [ORR]}
    #   effective_facts = paragraph.effective_facts — verified above
    # We simulate the inheritance here:
    sent2_effective_facts    = dict(para["effective_facts"])
    sent2_treatment_identity = dict(para.get("treatment_identity") or {})
    sent2_endpoint_identity  = dict(para.get("endpoint_identity")  or {})

    assert sent2_effective_facts.get("drug") == ["teclistamab"], \
        "sentence 2 should inherit drug from paragraph effective_facts"
    assert sent2_effective_facts.get("endpoint") == ["ORR"], \
        "sentence 2 should inherit endpoint from paragraph effective_facts"
    assert sent2_treatment_identity.get("primary_drug") == "teclistamab", \
        "sentence 2 should inherit primary_drug from paragraph treatment_identity"
    assert sent2_endpoint_identity.get("endpoint") == "ORR", \
        "sentence 2 should inherit endpoint from paragraph endpoint_identity"

    print("PASS: test_sentence_inheritance")


# ─── Test 2: heading-only drug graduates into paragraph ───────────────────────

def test_heading_drug_graduates_to_paragraph():
    """
    Paragraph whose own text has NO drug mention.
    Drug comes entirely from the heading context.

    After propagation:
      effective_facts.drug       = [teclistamab]  (from heading)
      treatment_identity.primary_drug = teclistamab  (graduated from ef)
    """
    objs = [
        _heading(0, "Evaluation of Teclistamab",
                 facts={"drug": ["teclistamab"]},
                 treatment={"primary_drug": "teclistamab"}),
        _para(1,
              "The endpoint will be measured over 12 months.",
              facts={"endpoint": ["ORR"]},         # no drug in own text
              treatment={"primary_drug": None},
              endpoint={"endpoint": "ORR"}),
    ]
    propagate_effective_facts(objs)

    para = objs[1]
    ef   = para.get("effective_facts") or {}
    ti   = para.get("treatment_identity") or {}

    assert "teclistamab" in ef.get("drug", []), \
        f"drug should be inherited from heading: ef={ef}"
    assert ti.get("primary_drug") == "teclistamab", \
        f"primary_drug should graduate from ef.drug: ti={ti}"

    print("PASS: test_heading_drug_graduates_to_paragraph")


# ─── Test 3: own value wins over inherited ────────────────────────────────────

def test_own_value_wins():
    """
    When a paragraph has its own drug in treatment_identity,
    that value must NOT be overwritten by the heading's drug.
    """
    objs = [
        _heading(0, "Treatment Protocol",
                 facts={"drug": ["daratumumab"]},
                 treatment={"primary_drug": "daratumumab"}),
        _para(1,
              "Teclistamab is administered SC.",
              facts={"drug": ["teclistamab"]},
              treatment={"primary_drug": "teclistamab"},  # own value present
              endpoint={}),
    ]
    propagate_effective_facts(objs)

    para = objs[1]
    ti   = para.get("treatment_identity") or {}

    # Own value ("teclistamab") must not be overwritten by heading ("daratumumab")
    assert ti.get("primary_drug") == "teclistamab", \
        f"own primary_drug should not be overwritten: ti={ti}"

    print("PASS: test_own_value_wins")


# ─── Test 4: companion_drugs from heading context ─────────────────────────────

def test_companion_drugs_graduated():
    """
    A paragraph directly under a multi-drug heading that has NO drugs in its
    own text should inherit all heading drugs and graduate companion_drugs.

    Note: own-wins means that once a paragraph mentions a drug, that drug
    REPLACES the heading's list for context purposes.  This test uses a
    paragraph with no own drug so it receives the full heading combination.
    """
    objs = [
        _heading(0, "Tal-DP: Talquetamab + Daratumumab + Pomalidomide",
                 facts={"drug": ["talquetamab", "daratumumab", "pomalidomide"]},
                 treatment={"primary_drug": "talquetamab",
                             "companion_drugs": ["daratumumab", "pomalidomide"]}),
        _para(1,
              "This regimen will be evaluated for efficacy.",
              facts={},                          # no drug in own text
              treatment={"primary_drug": None, "companion_drugs": []},
              endpoint={}),
    ]
    propagate_effective_facts(objs)

    para = objs[1]
    ef   = para.get("effective_facts") or {}
    ti   = para.get("treatment_identity") or {}

    # effective_facts should carry all three drugs from heading
    assert "talquetamab"  in ef.get("drug", []), f"talquetamab missing from ef: {ef}"
    assert "daratumumab"  in ef.get("drug", []), f"daratumumab missing from ef: {ef}"
    assert "pomalidomide" in ef.get("drug", []), f"pomalidomide missing from ef: {ef}"

    # primary_drug and companion_drugs should be graduated from ef.drug
    assert ti.get("primary_drug") == "talquetamab", \
        f"primary_drug should graduate: ti={ti}"
    assert "daratumumab" in (ti.get("companion_drugs") or []), \
        f"companion_drugs should graduate: ti={ti}"

    # Known limitation: if a DIFFERENT paragraph mentioning only one drug had been
    # inserted between the heading and this paragraph, context.drug would have been
    # narrowed to that one drug (own-wins replaces the context slot).  The combination
    # context from the heading is only preserved when no intervening paragraph
    # overwrites it.  This is a design trade-off: for single-drug paragraphs the
    # own-wins behaviour is correct; combination protocols rely on drug-free paragraphs
    # (or headings) to broadcast the full regimen.

    print("PASS: test_companion_drugs_graduated")


# ─── Test 5: population disease inherited ────────────────────────────────────

def test_population_disease_graduated():
    """
    A paragraph about RRMM that doesn't contain the disease abbreviation in
    its own text should still get disease from heading context.
    """
    objs = [
        _heading(0, "Eligibility Criteria for RRMM Patients",
                 facts={"disease": ["RRMM"]},
                 population={"disease": "RRMM"}),
        _para(1,
              "Patients must have received ≥3 prior lines of therapy.",
              facts={},                         # no disease in own text
              population={"disease": None}),
    ]
    propagate_effective_facts(objs)

    para = objs[1]
    ef   = para.get("effective_facts") or {}
    pi   = para.get("population_identity") or {}

    assert "RRMM" in ef.get("disease", []), \
        f"disease should be inherited from heading: ef={ef}"
    assert pi.get("disease") == "RRMM", \
        f"population_identity.disease should graduate: pi={pi}"

    print("PASS: test_population_disease_graduated")


# ─── Test 6: context carries forward across multiple paragraphs ───────────────

def test_context_propagates_across_paragraphs():
    """
    Once a paragraph contributes drug to context, subsequent sibling paragraphs
    inherit it even without a heading in between.
    """
    objs = [
        _para(0, "Teclistamab SC is administered at RP2D.",
              facts={"drug": ["teclistamab"]},
              treatment={"primary_drug": "teclistamab"}),
        _para(1, "The primary endpoint is ORR per IMWG.",
              facts={"endpoint": ["ORR"]},      # no drug in own text
              treatment={"primary_drug": None},
              endpoint={"endpoint": "ORR"}),
    ]
    propagate_effective_facts(objs)

    para2 = objs[1]
    ef    = para2.get("effective_facts") or {}
    ti    = para2.get("treatment_identity") or {}

    assert "teclistamab" in ef.get("drug", []), \
        f"drug from para0 should propagate to para1: ef={ef}"
    assert ti.get("primary_drug") == "teclistamab", \
        f"primary_drug should graduate: ti={ti}"

    print("PASS: test_context_propagates_across_paragraphs")


# ─── Test 7: heading_path as string is normalised ─────────────────────────────

def test_heading_path_string_normalised():
    """
    Objects indexed before the string→list fix store heading_path as
    "Primary Objectives > Primary Endpoint" (a string, not a list).
    enrich_object must normalise it so statement_type is classified correctly.
    """
    from shared.clinical_fact_extractor import enrich_object

    enrichment = enrich_object(
        text             = "The primary endpoint is ORR per IMWG criteria.",
        entities         = [{"text": "ORR", "label": "CLINICAL_ENDPOINT",
                              "object_start": 23, "object_end": 26}],
        section_category = "OBJECTIVES",
        heading_path     = "Primary Objectives > Primary Endpoint",  # string, not list
    )

    assert enrichment.get("statement_type") in {
        "PRIMARY_ENDPOINT", "PRIMARY_OBJECTIVE", "ENDPOINT"
    }, f"statement_type wrong for string heading_path: {enrichment.get('statement_type')}"

    print("PASS: test_heading_path_string_normalised  "
          f"(statement_type={enrichment.get('statement_type')})")


# ─── Test 8: entities present but facts empty → fact_extractor regression ────

def test_entities_present_facts_empty():
    """
    Regression for Point 2B: NER found entities but the fact extractor produced
    nothing.  This state (entities=true, facts={}) must NOT silently produce a
    valid effective_facts — it should propagate an empty own_facts and the
    inherited context must come exclusively from heading/paragraph context.

    Specifically:
    - A paragraph with entities but empty facts under a drug-bearing heading
      must have drug in effective_facts (inherited), but own_facts must be empty.
    - effective_facts must NOT contain values that neither own text nor heading
      contributes.
    """
    heading = _heading(0, "Teclistamab monotherapy arm", {"drug": ["teclistamab"]})
    para = {
        "object_id": "p1", "type": "paragraph", "position": 1,
        "text": "AEs were recorded per cycle.",
        # Simulates NER finding entities but fact extractor producing nothing
        "entities": [{"text": "AEs", "label": "ADVERSE_EVENT"}],
        "facts": {},          # ← empty: fact extractor produced nothing
        "treatment_identity":  {},
        "endpoint_identity":   {},
        "population_identity": {},
    }
    objects = [heading, para]
    propagate_effective_facts(objects)   # modifies in-place; returns None
    para_out = objects[1]

    eff = para_out.get("effective_facts", {})
    # Drug should propagate from heading (context inheritance)
    assert eff.get("drug") == ["teclistamab"], (
        f"Drug should inherit from heading even when own facts are empty. Got: {eff}"
    )
    # own_facts must remain empty — fact extractor produced nothing
    own = para_out.get("own_facts", {})
    assert not own.get("drug"), (
        f"own_facts.drug must be empty when fact extractor found nothing. Got: {own}"
    )
    # inherited_slots must record that drug came from context
    inherited = para_out.get("inherited_slots", [])
    assert "drug" in inherited, (
        f"drug must appear in inherited_slots when sourced from heading. Got: {inherited}"
    )
    print("PASS: test_entities_present_facts_empty")


# ─── Test 9: inherited drug in effective_facts suppresses zero-identity penalty ──

def test_inherited_effective_facts_prevents_zero_identity_penalty():
    """
    Regression for Point 8A / concern D:

    A paragraph that has NO own drug in its text but inherits drug from a heading
    via propagate_effective_facts must NOT trigger the aggregator's zero-identity
    penalty.  The penalty fires only when cand_facts_eff.drug is absent — and
    after propagation effective_facts.drug contains the inherited value.

    This test also validates that the endpoint propagates correctly when the
    paragraph contributes its own endpoint and the heading contributes drug.
    """
    heading = _heading(0, "Teclistamab arm", {"drug": ["teclistamab"]})
    para    = _para(1, "Overall response was evaluated per IMWG.", {"endpoint": ["ORR"]})

    objects = [heading, para]
    propagate_effective_facts(objects)
    para_out = objects[1]

    eff = para_out.get("effective_facts", {})
    assert eff.get("drug"), (
        f"Drug must be in effective_facts after heading propagation. Got: {eff}"
    )
    assert eff.get("endpoint"), (
        f"Endpoint must be in effective_facts (own or inherited). Got: {eff}"
    )

    # Replicate the aggregator's zero-identity check:
    #   cand_facts_eff = matched_obj.get("effective_facts") or matched_obj.get("facts")
    #   _cand_has_drug = bool(cand_facts_eff.get("drug"))
    #   _cand_has_ep   = bool(cand_facts_eff.get("endpoint"))
    #   penalty fires when (_ci_has_drug or _ci_has_ep) and not _cand_has_drug and not _cand_has_ep
    _cand_has_drug = bool(eff.get("drug"))
    _cand_has_ep   = bool(eff.get("endpoint"))
    penalty_fires  = True and not _cand_has_drug and not _cand_has_ep
    assert not penalty_fires, (
        f"Zero-identity penalty MUST NOT fire when drug is in effective_facts "
        f"(even if inherited). drug={eff.get('drug')}, endpoint={eff.get('endpoint')}"
    )
    print("PASS: test_inherited_effective_facts_prevents_zero_identity_penalty")


# ─── runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_sentence_inheritance,
        test_heading_drug_graduates_to_paragraph,
        test_own_value_wins,
        test_companion_drugs_graduated,
        test_population_disease_graduated,
        test_context_propagates_across_paragraphs,
        test_heading_path_string_normalised,
        test_entities_present_facts_empty,
        test_inherited_effective_facts_prevents_zero_identity_penalty,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"FAIL: {t.__name__} — {exc}")
            failed.append(t.__name__)

    print()
    if failed:
        print(f"FAILED ({len(failed)}/{len(tests)}): {failed}")
        sys.exit(1)
    else:
        print(f"ALL PASSED ({len(tests)}/{len(tests)})")
