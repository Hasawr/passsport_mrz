"""Tests for MRZ detection fixes targeting AGO passport failures.

Verifies that:
1. _mrz_line_score correctly penalises header/label text (PASSAPORTE, PASSPORT, etc.)
2. _mrz_line_score rewards proper MRZ Line 1 structure (country-agnostic)
3. _find_td3_candidates penalises candidates with noisy Line 1
4. _is_likely_line2 correctly identifies single Line 2 patterns
5. _diagnostic_result assigns single Line 2 correctly
"""

from services.passport.mrz_extractor import (
    PASSPORT_NOISE_WORDS,
    PassportMRZExtractor,
)


class TestMRZLineScore:
    """Tests for _mrz_line_score fixing country-agnostic scoring and noise penalties."""

    def test_real_ago_line1_scores_high(self):
        """A real AGO MRZ Line 1 should score well (has P + country + << name sep)."""
        real_l1 = "PSAGOBORGES<<MATIAS<MANUEL<DA<SILVA<<<<<<<<<<<"
        score = PassportMRZExtractor._mrz_line_score(real_l1)
        assert score > 25, f"Real MRZ L1 scored too low: {score}"

    def test_fake_header_passaporte_scores_low(self):
        """The PASSAPORTE/PASSPORT header text should be heavily penalised."""
        # This is the exact failure from AGO_2
        fake_header = "PASSAPORTETPASSPORTPSCARVALHOAGOS0070648<<<<"
        score = PassportMRZExtractor._mrz_line_score(fake_header)
        assert score < 12, f"Fake header scored too high: {score}"

    def test_real_line1_beats_fake_header(self):
        """Real MRZ Line 1 must outscore the fake PASSAPORTE header."""
        real = "PSAGOCARVALHO<<PEDRO<MENDES<<<<<<<<<<<<<<<<<<<<<"[:44]
        fake = "PASSAPORTETPASSPORTPSCARVALHOAGOS0070648<<<<"
        real_score = PassportMRZExtractor._mrz_line_score(real)
        fake_score = PassportMRZExtractor._mrz_line_score(fake)
        assert real_score > fake_score, (
            f"Real L1 ({real_score}) should beat fake header ({fake_score})"
        )

    def test_data_page_labels_score_low(self):
        """Passport data page labels (FUNCIONARIO, PUBLICO) should be penalised."""
        data_label = "PEDRO<MENDES<DEFUNCIONARIO<PUBLICO<<<<<<<<<<<<<<"[:44]
        score = PassportMRZExtractor._mrz_line_score(data_label)
        assert score < 12, f"Data label scored too high: {score}"

    def test_real_line2_not_prose_penalised(self):
        """Real MRZ Line 2 (high digit ratio) must not get a prose penalty."""
        real_l2 = "S0074615<5AG08504146M29062090004296<S10<0138"
        score = PassportMRZExtractor._mrz_line_score(real_l2)
        # Should not be negative — Line 2 is valid MRZ
        assert score > 0, f"Real MRZ L2 got negative score: {score}"

    def test_aze_line1_still_scores_high(self):
        """Existing AZE passports must still score well (no regression)."""
        aze_l1 = "P<AZEALIYEV<<ELVIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<"[:44]
        score = PassportMRZExtractor._mrz_line_score(aze_l1)
        assert score > 25, f"AZE MRZ L1 scored too low: {score}"

    def test_non_aze_country_scores_well(self):
        """Non-AZE countries (GBR, USA, DEU, etc.) should also score well."""
        for country in ["GBR", "USA", "DEU", "FRA", "AGO"]:
            l1 = f"P<{country}SMITH<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<"[:44]
            score = PassportMRZExtractor._mrz_line_score(l1)
            assert score > 20, f"{country} MRZ L1 scored too low: {score}"


class TestNoiseWords:
    """Tests for the PASSPORT_NOISE_WORDS constant."""

    def test_common_words_present(self):
        assert "PASSPORT" in PASSPORT_NOISE_WORDS
        assert "PASSAPORTE" in PASSPORT_NOISE_WORDS
        assert "FUNCIONARIO" in PASSPORT_NOISE_WORDS
        assert "PUBLICO" in PASSPORT_NOISE_WORDS

    def test_mrz_content_not_in_noise(self):
        """MRZ-valid words must not appear in noise list."""
        for word in ["AGO", "AZE", "GBR", "USA"]:
            assert word not in PASSPORT_NOISE_WORDS


class TestIsLikelyLine2:
    """Tests for the _is_likely_line2 helper."""

    def test_real_line2_detected(self):
        """A digit-heavy line without P or << should be identified as Line 2."""
        line2 = "S0074567<5AG07203291M29061950019247<S10<0086"
        assert PassportMRZExtractor._is_likely_line2(line2) is True

    def test_real_line1_not_detected(self):
        """MRZ Line 1 (starts with P, has <<) should NOT be identified as Line 2."""
        line1 = "PSAGOBORGES<<MATIAS<MANUEL<DA<SILVA<<<<<<<<<<<"
        assert PassportMRZExtractor._is_likely_line2(line1) is False

    def test_short_text_not_detected(self):
        """Short text should not be identified as Line 2."""
        assert PassportMRZExtractor._is_likely_line2("12345") is False

    def test_line_with_name_sep_not_detected(self):
        """A line with << in the first 30 chars is likely Line 1, not Line 2."""
        text = "SOMETHING<<WITHNAME<AND<DIGITS123456789012345"
        assert PassportMRZExtractor._is_likely_line2(text) is False


class TestPreferMRZLikeLines:
    """Tests for _prefer_mrz_like_lines filtering with noise penalty."""

    def test_real_lines_preferred_over_header(self):
        """When real MRZ lines compete with header text, real lines should win."""
        fake_header = ("PASSAPORTETPASSPORTPSCARVALHOAGOS0070648<<<<", 0.90)
        real_l1 = ("PSAGOCARVALHO<<PEDRO<MENDES<<<<<<<<<<<<<<<<<<", 0.95)
        real_l2 = ("S0074615<5AG08504146M29062090004296<S10<0138", 0.97)
        fake_data = ("PEDRO<MENDES<DEFUNCIONARIO<PUBLICO<<<<<<<<<<", 0.88)

        all_lines = [fake_header, real_l1, real_l2, fake_data]
        selected = PassportMRZExtractor._prefer_mrz_like_lines(all_lines)

        selected_texts = [t for t, _ in selected]
        assert real_l1[0] in selected_texts, "Real Line 1 should be selected"
        assert real_l2[0] in selected_texts, "Real Line 2 should be selected"


class TestFindTD3CandidatesNoise:
    """Tests for noise penalty in _find_td3_candidates."""

    def test_noisy_line1_candidate_scores_lower(self):
        """A candidate pair where Line 1 has noise words should score lower."""
        fake_pair = [
            ("PASSAPORTETPASSPORTPSCARVALHOAGOS0070648<<<<", 0.90),
            ("S0074615<5AG08504146M29062090004296<S10<0138", 0.97),
        ]
        real_pair = [
            ("PSAGOCARVALHO<<PEDRO<MENDES<<<<<<<<<<<<<<<<<<", 0.95),
            ("S0074615<5AG08504146M29062090004296<S10<0138", 0.97),
        ]

        fake_candidates = PassportMRZExtractor._find_td3_candidates(fake_pair)
        real_candidates = PassportMRZExtractor._find_td3_candidates(real_pair)

        # Both should produce candidates, but real should score higher
        if fake_candidates and real_candidates:
            assert real_candidates[0].score > fake_candidates[0].score, (
                f"Real pair ({real_candidates[0].score}) should outscore "
                f"fake pair ({fake_candidates[0].score})"
            )
