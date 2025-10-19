PANEL_PARSE_LEXICON = {
  # -----------------------------
  # HEADER / TOP-SECTION FIELDS
  # -----------------------------
  "header": {
    # Panel display name (skip if it looks like an existing panel)
    "panel_name": {
      "labels": [
        "panel", "panelboard", "panel schedule", "pnl", "pnl.", "pnl schedule", "panelboard schedule"
      ],
      # Pull the token after the label: e.g., "Panel: LP", "PANEL LP", "Panelboard Schedule - PP1"
      "name_patterns": [
        r"\bPANEL(?:BOARD)?\s+SCHEDULE\s*[-:]\s*([A-Z0-9._\-]{1,32})",
        r"\bPANEL(?:BOARD)?\s*[:\-]\s*([A-Z0-9._\-]{1,32})",
        r"\bPNL\.?\s*[:\-]?\s*([A-Z0-9._\-]{1,32})",
        r"\bPANEL(?:BOARD)?\s+([A-Z0-9._\-]{1,32})"
      ],
      # Skip entirely if captured name is exactly one of these (case-insensitive)
      "skip_if_name_exact": ["MSB", "SWBD", "SWITCHBOARD"],
      # If any of these tokens appear near the captured name, treat this panel as "skip"
      "skip_if_name_contains": [
        "EXIST", "EXISTING", "DEMO", "DEMOLISH", "REMOVE", "FUTURE", "ABANDON", "NOT IN SCOPE"
      ],
      "max_name_len": 32
    },

    # System voltage; normalize to {120, 208, 480, 600}
    "voltage": {
      "labels": [
        "volts", "voltage", "system volts", "system voltage", "volts ph", "pnl volts", "pnl voltage"
      ],
      # Extract common forms; we’ll normalize later
      "value_patterns": [
        r"\b(\d{2,3})\s*/\s*(\d{2,3})\s*[Vv]?\b",           # 120/208, 120/240, etc.
        r"\b(\d{2,3})\s*[Yy]\s*/\s*(\d{2,3})\s*[Vv]?\b",    # 208Y/120V, 480Y/277
        r"\b(\d{2,3})\s*[Vv]\b",                            # 208V, 480V, 600V, 120V
        r"\b(\d{2,3})\b(?=\s*(?:V|VOLTS)\b)"                # 208 VOLTS
      ],
      "allowed": [120, 208, 480, 600]
    },

    # Panel bus rating (amperage of the bus / main lugs)
    "amperage": {
      "labels": [
        "amps", "amperage", "rating", "bus rating", "bus amps", "bus amperage", "lug amps", "bus", "bus a", "amps rating"
      ],
      "value_patterns": [
        r"\b(\d{2,4})\s*A\b",      # 225 A, 1200 A
        r"\b(\d{2,4})\b(?=\s*A\b)",# 225 (A)
        r"\b(?:BUS|LUGS?)\s*(?:RATING|AMPS?|AMPERAGE)?\s*[:\-]?\s*(\d{2,4})\s*A?\b",
        r"\bRATING\s*[:\-]?\s*(\d{2,4})\s*A?\b"
      ],
      "normalize": "int"          # strip trailing A/units; return number
    },

    # Main breaker amperage (MCB); if present drives main-breaker logic
    "main_breaker_amperage": {
      "labels": [
        "mcb", "main breaker", "breaker (main)", "main", "main amps",
        "mcb amps", "main breaker amps", "breaker amps", "mcb amperage",
        "main amperage", "main breaker amperage", "mains rating", "device", "main device", "device amps", "main device amps"
      ],
      # Prefer patterns that explicitly say MAIN/MCB/BREAKER near the number
      "value_patterns": [
        r"\b(?:MAIN|MCB)\s*(?:BREAKER)?\s*(?:RATING|AMPS?|AMPERAGE)?\s*[:\-]?\s*(\d{2,4})\s*A?\b",
        r"\bMAINS?\s*(?:BREAKER|RATING)?\s*(?:AMPS?|AMPERAGE)?\s*[:\-]?\s*(\d{2,4})\s*A?\b",
        r"\bBREAKER\s*(?:RATING|AMPS?|AMPERAGE)?\s*[:\-]?\s*(\d{2,4})\s*A?\b"
      ],
      "normalize": "int"
    },

    # Interrupting rating (AIC/SCCR); normalized to integer kA (e.g., 42)
    "int_rating": {
      "labels": [
        "a.i.c. rating", "aic rating", "aic", "a.i.c.", "int rating",
        "interrupting", "interrupting rating", "sccr", "int"
      ],
      "value_patterns": [
        r"\b(\d{2,3})\s*[Kk]\s*(?:A\.?I\.?C\.?|A|AIC)?\b",                     # 65k, 65 kAIC, 42 k
        r"\bA\.?I\.?C\.?\s*[:\-]?\s*(\d{2,3})\s*[Kk]?\b",                      # A.I.C: 42k
        r"\bSCCR\s*[:\-]?\s*(\d{2,3})\s*[Kk]?\b",                              # SCCR: 65k
        r"\bINT(?:ERRUPTING)?\s*RATING\s*[:\-]?\s*(\d{2,3})\s*[Kk]?\b",
        r"\b(\d{2,3})\s*[Kk][Aa](?:[Ii][Cc])?\b",                               # 42kA, 42kAIC
        # NEW: plain thousands near the AIC/SCCR text (e.g., "A.I.C. Rating: 35,000" or "35000")
        r"\b(?:A\.?I\.?C\.?|AIC|SCCR|INT(?:ERRUPTING)?(?:\s*RATING)?)\b[^\n\r]{0,20}?(\d{1,3}(?:,\d{3})+)\b",
        r"\b(?:A\.?I\.?C\.?|AIC|SCCR|INT(?:ERRUPTING)?(?:\s*RATING)?)\b[^\n\r]{0,20}?(\d{4,6})\b"
      ],
      "normalize": "int_kilo_or_thousands"  # '65k'->65, '35,000'/'35000'->35
    },

    # Surge protection (SPD/TVSS) size in kA; normalized integer (e.g., 160)
    "spd": {
      "labels": [
        "spd", "surge", "surge protection", "surge protection device",
        "tvss", "surge suppressor", "transient voltage surge suppressor"
      ],
      "value_patterns": [
        r"\b(?:SPD|SURGE|TVSS)[^\n\r]*?(\d{2,3})\s*[Kk][Aa]?\b",
        r"\b(\d{2,3})\s*[Kk][Aa]?\b(?=.*\b(?:SPD|SURGE|TVSS)\b)"
      ],
      "normalize": "int"  # treat the kA number as integer (e.g., 160)
    }
  },

  # -----------------------------
  # TABLE / BREAKER HEADERS
  # -----------------------------
  "table": {
    "columns": {
      # Left/right mirrored halves share the same header aliases
      "poles": [
        "poles", "cb poles", "c.b. poles", "bkr poles", "breaker poles", "pole", "p"
      ],
      "amps": [
        "amps", "amperage", "cb amps", "c.b. amps", "breaker amps", "bkr amps",
        "size (a)", "size a", "size", "rating (a)", "rating a", "trip", "trip amps", "trip amperage"
      ],
      # Some schedules put both in one field like "20/1", "30-2P", "1P-20A"
      "combo_amps_poles": [
        "bkr", "breaker", "cb", "device", "size/poles", "a/p", "p/a", "trip/poles"
      ],
      "description": [
        "description", "designation", "load", "load name", "load description",
        "bkr name", "breaker name", "breaker description", "remarks", "comments"
      ],
      "circuit": [
        "ckt", "circuit", "circuit #", "ckt #", "ckt no.", "circuit number", "circuit no", "circuit id"
      ]
    },

    # When a single field carries both numbers (per-row), detect with these:
    "combo_value_patterns": [
      r"\b(\d{1,3})\s*/\s*([123])\b",                # 20/1, 30/2, 100/3
      r"\b([123])\s*P(?:OLES?)?\s*[-/]?\s*(\d{1,3})\s*A?\b",  # 1P-20A, 2P/30
      r"\b(\d{1,3})\s*A?\s*[-/ ]\s*([123])\s*P\b",   # 20A-1P, 30-2P
      r"\b(\d{1,3})\s*[A]?\s*\/\s*([123])\s*[P]\b"   # 20/1P, 30/2P
    ],

    # Tokens that mean the row is intentionally empty (do NOT count a breaker):
    "empty_row_tokens": [
      "SPACE", "PROVISION", "EMPTY", "BLANK", "UNUSED", "OPEN"
    ],
    # Tokens that look empty but ARE valid devices (keep them):
    "valid_spare_tokens": ["SPARE", "SPARES"],
  }
}
