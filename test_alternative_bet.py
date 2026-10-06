import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "api_server", pathlib.Path(__file__).parent.parent / "api_server.py"
)
api_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_server)
generate_hint = api_server.generate_hint
generate_alternative_bet = api_server.generate_alternative_bet


def test_alternative_shows_btts_when_primary_is_straight_win():
    primary = generate_hint("Home", "Away", {"home_win": 0.70, "draw": 0.18, "away_win": 0.12},
                             btts_yes=0.40, btts_no=0.60)
    alt = generate_alternative_bet(primary, btts_yes=0.40, btts_no=0.60)
    assert alt == {"market": "BTTS", "selection": "No", "confidence": 0.6}


def test_alternative_is_none_when_primary_is_already_btts():
    primary = generate_hint("Home", "Away", {"home_win": 0.35, "draw": 0.28, "away_win": 0.37},
                             btts_yes=0.68, btts_no=0.32)
    assert primary["market"] == "BTTS"  # sanity check on the fixture itself
    alt = generate_alternative_bet(primary, btts_yes=0.68, btts_no=0.32)
    assert alt is None


def test_alternative_still_shows_when_no_primary_hint():
    primary = generate_hint("Home", "Away", {"home_win": 0.36, "draw": 0.32, "away_win": 0.32},
                             btts_yes=0.55, btts_no=0.45)
    assert primary is None
    alt = generate_alternative_bet(primary, btts_yes=0.55, btts_no=0.45)
    assert alt == {"market": "BTTS", "selection": "Yes", "confidence": 0.55}


def test_alternative_picks_whichever_btts_side_is_more_likely():
    alt_yes = generate_alternative_bet(None, btts_yes=0.61, btts_no=0.39)
    assert alt_yes["selection"] == "Yes"
    alt_no = generate_alternative_bet(None, btts_yes=0.39, btts_no=0.61)
    assert alt_no["selection"] == "No"
