"""RiichiEnv — four-player hanchan (East-South) riichi mahjong engine.

Default ruleset = Tenhou Houou / Majsoul Ouja standard:
- kuitan ON (open tanyao), aka dora ON (3 red fives)
- kuikae OFF (no swap-calling)
- kiriage mangan ON
- atamahane ON (head bump; double/triple ron OFF by default)
- nagashi mangan (configurable, default ON)
- double yakuman counting ON, kazoe yakuman capped at single yakuman

Wall layout (136 ids in draw order, MT19937 seeded shuffle):
- indices 0..121   live wall, drawn front to back
- indices 122..135 dead wall (wanpai):
  * 122..125 rinshan pool (4 replacement draws, taken back-to-front: 125,124,123,122)
  * 126..130 dora indicators (initial + one per kan, flipped 126,127,...,130)
  * 131..135 ura indicators (mirroring the dora indicators 1:1)

Public API:
  game = RiichiGame(config=RiichiConfig(), seed=42)
  obs   = game.state.get_observation(seat)  # docs/observation_schema.md §2 dict
  legal = game.state.legal_actions(seat)    # action mask of the current decision
  res   = game.step(action)                 # advance the game one decision

Action dicts:
  {"type":"discard","tile":136}   {"type":"riichi","tile":136}
  {"type":"chow","tiles":[...]}   {"type":"pon","tiles":[...]}
  {"type":"kan","tiles":[...]}    (ankan/shouminkan on own turn, daiminkan on a claim)
  {"type":"ron"}   {"type":"tsumo"}   {"type":"kyushu"}   {"type":"pass"}

Rule switches (RiichiConfig) — see class docstring.

Known limitations (documented choices):
- Temporary (same-go-around) furiten: declining ANY tile that completes the
  hand shape sets it, with or without yaku (passing or taking a different
  call); it clears on the player's own next draw. A riichi'd player who
  declines a legal ron also gets permanent riichi furiten.
- An ankan can NEVER be robbed (platform v1.7 / Tenhou); only a kakan
  (shouminkan) opens a chankan window, robbable by any winning hand.

mjai.app (Rust reference) parity notes — cross-validated by tests/test_oracle.py:
- A kan consumes one of the 70 draws: wall_left counts every draw including
  rinshan replacements (with K kans only 70-K live-wall tiles are drawn).
- Kan dora timing: ankan reveals immediately; daiminkan/kakan reveal at the
  kan player's next discard (or next tsumo if they kakan again first).
- Riichi may be declared while furiten or with dead waits (shape-only check).
- Double ron: the ronner closest to the discarder takes all riichi sticks and
  the honba bonus; the others collect the base payment only.
- Agari-yame at all-last needs return_score+ and (joint) top; West rounds
  follow mjai.app rules (renchan continues regardless of scores, hard cap
  after West-4), leftover riichi sticks go to the top player at game end.
"""
from __future__ import annotations

import itertools
import random
from collections import Counter
from dataclasses import dataclass, field

from mahjong.agari import Agari
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, HandConstants, OptionalRules
from mahjong.meld import Meld
from mahjong.shanten import Shanten

from riichi.tiles import (
    RED_FIVES,
    counts34,
    mjai_pai,
    parse_mjai_pai,
    is_terminal_or_honor,
    is_wind,
    kind_of,
    sort_tiles136,
    tiles136_to_str,
)


@dataclass
class RiichiConfig:
    """Optional rule switches. Defaults match Tenhou Houou / Majsoul Ouja."""

    aka: bool = True                # red fives in the wall (fixed ids 16/52/88)
    kuitan: bool = True             # open tanyao allowed
    kuikae: bool = False            # False = kuikae (swap calling) FORBIDDEN
    kiriage_mangan: bool = False    # platform v1.6 (Tenhou): no kiriage
    atamahane: bool = True          # head bump (only when double_ron=False)
    double_ron: bool = True         # platform v1.6 (Tenhou): double ron allowed
                                    # (overrides atamahane); triple ron = ron3
                                    # abortive draw
    nagashi_mangan: bool = True     # exhaustive-draw nagashi mangan
    kyushu_kyuhai: bool = True      # nine-terminal hand declaration
    four_winds_ryuukyoku: bool = True
    four_kans_ryuukyoku: bool = True
    chankan_kokushi: bool = False   # reserved for future extensions; as of
                                    # platform v1.7 an ankan can NEVER be
                                    # robbed (the engine ignores this flag)
    kazoe_limit: int = HandConstants.KAZOE_LIMITED
    agari_yame: bool = True         # dealer win in South-4 ends the game
    west_extension: bool = True     # Tenhou Houou / Majsoul Ouja: continue
                                    # into West rounds while every score is
                                    # still below return_score after South-4
    start_score: int = 25000
    return_score: int = 30000
    double_yakuman: bool = False    # platform v1.6 (Tenhou): all yakuman are
                                    # single (suuankou-tanki / kokushi-13
                                    # included); set True for Majsoul rules


class _StateView:
    """Exposes schema-oriented read API: state.get_observation(seat)."""

    def __init__(self, game: "RiichiGame") -> None:
        self._game = game

    def get_observation(self, seat=None, include_explain: bool = False) -> dict:
        return self._game.get_observation(seat, include_explain)

    def legal_actions(self, seat=None) -> dict:
        return self._game.legal_actions(seat)


def _empty_legal() -> dict:
    # "kyushu" is an engine extension key (not part of schema §2)
    return {"discard": [], "riichi": [], "chow": [], "pon": [], "kan": [],
            "ron": False, "tsumo": False, "kyushu": False}


class RiichiGame:
    """Four-player hanchan riichi engine (see module docstring)."""

    WINDS_34 = (27, 28, 29, 30)

    def __init__(self, config: RiichiConfig | None = None, seed: int | None = None) -> None:
        self.cfg = config if config is not None else RiichiConfig()
        self.seed = seed if seed is not None else 0
        self.rng = random.Random(self.seed)
        self.game_id = "engine-%d-%d" % (self.seed, id(self) % 100000)
        self.state = _StateView(self)
        self.record_decisions = False
        self.records: list = []
        self.events: list = []
        self.scores = [self.cfg.start_score] * 4
        self.dealer = 0
        self.round_idx = 0
        self.honba = 0
        self.kyoutaku = 0          # riichi sticks on the table
        self.round_result = None
        self.game_result = None
        self._game_no = 0
        self._new_round()

    # ------------------------------------------------------------------ setup

    def _emit_start_kyoku(self) -> None:
        self._emit({"type": "start_kyoku",
                    "bakaze": "ESW"[min(self.round_idx // 4, 2)],
                    "kyoku": self.round_idx + 1, "honba": self.honba,
                    "kyotaku": self.kyoutaku, "oya": self.dealer,
                    "scores": list(self.scores),
                    "dora_marker": mjai_pai(self.wall[126]),
                    "tehais": [[mjai_pai(t) for t in h] for h in self.hands]})
    def _new_round(self) -> None:
        self._game_no += 1
        wall = list(range(136))
        self.rng.shuffle(wall)
        self.wall = wall
        self.draw_ptr = 0          # live wall draw pointer (0..121)
        self.rinshan_ptr = 125     # rinshan pool pointer (drawn back-to-front)
        self.wall_left = 70        # remaining drawable tiles (70 at round start)
        self.dora_indicators = [wall[126]]
        self.ura_indicators = [wall[131]]
        self.n_kan = 0
        self.kan_owners: list = []
        self.hands = [[] for _ in range(4)]
        self.melds = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.riichi_declared = [False] * 4
        self.riichi_furiten = [False] * 4     # permanent furiten after riichi pass
        self.temp_furiten = [False] * 4       # same-go-around furiten
        self._riichi_draws = [0] * 4          # own draws since riichi
        self._ippatsu_broken = [False] * 4
        self._daburu = [False] * 4
        self._has_drawn = [False] * 4
        self._draws_made = [0] * 4
        self._calls_made = False
        self._first_discards: list = []
        self._draw_counter = 0
        self.phase = "draw"        # draw | claim | chankan | round_end | game_end
        self.turn = self.dealer
        self.claim_queue: list = []
        self.chankan_queue: list = []
        self._shape_seers: list = []
        self._claim_tile = -1
        self._claim_discarder = -1
        self._ronners: list = []
        self._pao: dict = {}
        self._kan_pending = None
        self.drawn_tile = -1
        self._rinshan_pending = False
        self._rinshan_current = False
        self._dora_at_tsumo = None      # kan dora revealed at next tsumo
        self._dora_at_discard = None    # kan dora revealed at next discard
        self._haitei_pending = False
        self._houtei_pending = False
        self._kuikae_kind = -1
        self._last_discard_riichi = False
        self.last_event = {"type": "deal", "seat": -1, "tile": -1}
        # deal 13 tiles to each player
        for seat in range(4):
            hand = self.wall[self.draw_ptr:self.draw_ptr + 13]
            self.draw_ptr += 13
            self.hands[seat] = sort_tiles136(hand)
        self._emit_start_kyoku()
        self._enter_draw(self.dealer)

    # ------------------------------------------------------------- turn engine

    def _seats_from(self, seat: int) -> list:
        return [(seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4]

    def _emit(self, ev: dict) -> None:
        ev = dict(ev)
        ev.setdefault("round", self.round_idx)
        ev.setdefault("honba", self.honba)
        self.events.append(ev)

    def _enter_draw(self, seat: int) -> None:
        """Draw a tile for seat and put the game into the draw (discard) phase."""
        if self.wall_left <= 0:
            self._ryuukyoku("exhaustive")
            return
        rinshan = self._rinshan_pending
        self._rinshan_pending = False
        self._rinshan_current = rinshan
        if self._dora_at_tsumo is not None:
            self._flip_dora_indicator(self._dora_at_tsumo)
            self._dora_at_tsumo = None
        if rinshan:
            tile = self.wall[self.rinshan_ptr]
            self.rinshan_ptr -= 1
        else:
            tile = self.wall[self.draw_ptr]
            self.draw_ptr += 1
        self.wall_left -= 1
        self.drawn_tile = tile
        self.turn = seat
        self.phase = "draw"
        self._draw_counter += 1
        self._has_drawn[seat] = True
        self._draws_made[seat] += 1
        if self.riichi_declared[seat]:
            self._riichi_draws[seat] += 1
        self.temp_furiten[seat] = False
        self._kuikae_kind = -1
        self._haitei_pending = (self.wall_left == 0) and not rinshan
        self.hands[seat].append(tile)
        self.hands[seat] = sort_tiles136(self.hands[seat])
        self.last_event = {"type": "draw", "seat": seat, "tile": tile}
        self._emit({"type": "tsumo", "actor": seat,
                    "pai": mjai_pai(tile),
                    "seat": seat, "tile": tile, "rinshan": rinshan})

    # ------------------------------------------------------------ legal actions

    def legal_actions(self, seat=None) -> dict:
        seat = self.turn if seat is None else seat
        la = _empty_legal()
        if self.phase in ("round_end", "game_end"):
            return la
        if self.phase == "draw":
            if seat != self.turn:
                return la
            self._fill_draw_legal(seat, la)
        elif self.phase == "claim" and self.claim_queue:
            item = self.claim_queue[0]
            if seat != item["seat"]:
                return la
            if item["kind"] == "ron":
                la["ron"] = True
            elif item["kind"] == "daiminkan":
                la["kan"] = [{"tiles": list(o)} for o in self._daiminkan_options(seat, self._claim_tile)]
            elif item["kind"] == "pon":
                la["pon"] = [{"tiles": list(o)} for o in self._pon_options(seat, self._claim_tile)]
            elif item["kind"] == "chow":
                la["chow"] = [{"tiles": list(o)} for o in self._chow_options(seat, self._claim_tile)]
        elif self.phase == "chankan" and self.chankan_queue:
            if seat == self.chankan_queue[0]["seat"]:
                la["ron"] = True
        return la

    def _fill_draw_legal(self, seat: int, la: dict) -> None:
        hand = self.hands[seat]
        if self._can_tsumo(seat):
            la["tsumo"] = True
        # discards
        if self.riichi_declared[seat]:
            discards = [self.drawn_tile] if self.drawn_tile >= 0 else []
        elif self.cfg.kuikae:
            discards = list(hand)
        else:
            # kuikae forbidden: cannot discard the kind that was just called
            discards = [t for t in hand if kind_of(t) != self._kuikae_kind]
        la["discard"] = discards
        # riichi
        la["riichi"] = self._riichi_options(seat)
        # kan (ankan + shouminkan)
        la["kan"] = [{"tiles": list(o)} for o in self._self_kan_options(seat)]
        # kyushu (extension key, not part of schema §2)
        if self._kyushu_legal(seat):
            la["kyushu"] = True

    # ---------------------------------------------------- shape / win queries

    def _concealed_counts(self, seat: int, hand=None) -> list:
        """34-counts of the concealed part incl. closed-kan tiles (not open melds)."""
        if hand is None:
            hand = self.hands[seat]
        counts = counts34(hand)
        for m in self.melds[seat]:
            if m["type"] == "kan" and not m["open"]:
                for t in m["tiles"]:
                    counts[t // 4] += 1
        return counts

    def _meld_sets_for_agari(self, seat: int) -> list:
        """All melds as kind lists to subtract in the Agari check:
        open melds contribute 3 kinds, closed kans contribute 4."""
        sets = []
        for m in self.melds[seat]:
            if m["open"]:
                sets.append([t // 4 for t in m["tiles"][:3]])
            else:
                sets.append([m["tiles"][0] // 4] * 4)
        return sets

    def _winning_kinds(self, seat: int, hand) -> set:
        """Kinds that complete the 13-tile hand (shape only, no yaku)."""
        if len(self.melds[seat]) == 4 and len(hand) == 1:
            # 4 组副露 + 手牌 1 张 = 单骑听牌（天凤判定：听手牌那张，无条件；
            #   oracle-verified: 53318fba S4, 4组副露+4p 单骑天凤判听）
            return {hand[0] // 4}
        hand_counts = counts34(hand)
        sets = self._meld_sets_for_agari(seat)
        counts = list(hand_counts)
        for ms in sets:
            for k in ms:
                counts[k] += 1
        out = set()
        for k in range(34):
            if hand_counts[k] >= 4:
                # 4 张同牌全在手牌 → 听第 5 张不存在，天凤判不听
                # （oracle-verified: 1d7e7319 E4, 3s×4 流局判不听；
                #  有副露的 4 张（副露3+手牌1）仍可听，53318fba S4）
                continue
            c = list(counts)
            c[k] += 1
            if Agari.is_agari(c, sets or None):
                out.add(k)
        return out

    def _discard_kinds(self, seat: int) -> set:
        return {d["tile"] // 4 for d in self.discards[seat]}

    def _is_furiten(self, seat: int, tile_kind: int) -> bool:
        """All three furiten types checked against the current 13-tile hand."""
        if self.temp_furiten[seat] or self.riichi_furiten[seat]:
            return True
        wait = self._winning_kinds(seat, self.hands[seat])
        if not wait:
            return True  # not tenpai -> cannot ron anyway
        return bool(wait & self._discard_kinds(seat))

    def _meld_objects(self, seat: int) -> list:
        out = []
        for m in self.melds[seat]:
            mtype = {"chi": Meld.CHI, "pon": Meld.PON, "kan": Meld.KAN,
                     "shouminkan": Meld.SHOUMINKAN}[m["type"]]
            out.append(Meld(meld_type=mtype, tiles=tuple(m["tiles"]), opened=m["open"],
                            called_tile=m.get("called"), who=seat,
                            from_who=m["from"] if m["from"] >= 0 else None))
        return out

    def _hand_config(self, seat: int, flags: dict) -> HandConfig:
        is_daburu = flags.get("daburu", False)
        is_riichi = self.riichi_declared[seat] and not is_daburu
        round_wind = 27 + min(self.round_idx // 4, 2)
        options = OptionalRules(
            has_open_tanyao=self.cfg.kuitan,
            has_aka_dora=self.cfg.aka,
            kiriage=self.cfg.kiriage_mangan,
            has_double_yakuman=self.cfg.double_yakuman,
            kazoe_limit=self.cfg.kazoe_limit,
        )
        return HandConfig(
            is_tsumo=flags.get("tsumo", False),
            is_riichi=is_riichi,
            is_ippatsu=flags.get("ippatsu", False),
            is_rinshan=flags.get("rinshan", False),
            is_chankan=flags.get("chankan", False),
            is_haitei=flags.get("haitei", False),
            is_houtei=flags.get("houtei", False),
            is_daburu_riichi=is_daburu,
            is_tenhou=flags.get("tenhou", False),
            is_chiihou=flags.get("chiihou", False),
            player_wind=27 + (seat - self.dealer) % 4,
            round_wind=round_wind,
            kyoutaku_number=0,
            tsumi_number=self.honba,
            options=options,
        )

    def _ippatsu_ok(self, seat: int, tsumo: bool) -> bool:
        if not self.riichi_declared[seat]:
            return False
        if self._ippatsu_broken[seat]:
            return False
        if tsumo:
            return self._riichi_draws[seat] == 1
        return self._riichi_draws[seat] == 0

    def _evaluate_win(self, seat: int, win_tile: int, tsumo: bool,
                      chankan: bool = False, houtei: bool = False,
                      haitei: bool = False, rinshan: bool = False,
                      tenhou: bool = False, chiihou: bool = False,
                      daburu: bool = False):
        """Full yaku/han/fu/cost evaluation. None when the hand cannot win."""
        meld_objs = self._meld_objects(seat)
        tiles = list(self.hands[seat])
        for m in self.melds[seat]:
            tiles.extend(m["tiles"])
        if not tsumo:
            tiles.append(win_tile)
        tiles = sort_tiles136(tiles)
        flags = {
            "tsumo": tsumo, "chankan": chankan, "houtei": houtei,
            "haitei": haitei, "rinshan": rinshan, "tenhou": tenhou,
            "chiihou": chiihou, "daburu": daburu,
            "ippatsu": self._ippatsu_ok(seat, tsumo),
        }
        config = self._hand_config(seat, flags)
        ura = None
        if self.riichi_declared[seat]:
            ura = [self.wall[131 + i] for i in range(len(self.dora_indicators))]
        try:
            result = HandCalculator.estimate_hand_value(
                tiles, win_tile, melds=meld_objs,
                dora_indicators=self.dora_indicators,
                config=config, ura_dora_indicators=ura)
        except ValueError:
            return None
        if result.error:
            return None
        yaku_names = [type(y).__name__ for y in (result.yaku or [])]
        if chankan and not any("Chankan" in y for y in yaku_names):
            # the mahjong lib skips Chankan on the kokushi (yakuman) path
            yaku_names.append("Chankan")
        han = result.han
        fu = result.fu
        cost = result.cost
        # 高点法（重新设计）：按拆法枚举，每法独立计算【该拆法的番 + 该拆法
        # 的符】→ 查点数表得打点 → 取打点最高者。番与符永远来自同一拆法
        # （oracle 实证：333p444p555p456sNN 荣和3p，一杯口拆法 3番40符
        #  > 刻子拆法 2番50符；lib 的番/符可能来自不同拆法导致不一致）。
        _YAKUMAN = {"Kokushi", "Suuankou", "Daisangen", "Shousuushii",
                    "Daisuushii", "Tsuuiisou", "Chinroutou", "Ryuuiisou",
                    "Chuurenpoutou", "KokushiMusou", "Suukantsu"}
        if han < 13 and not any("Yakuman" in y or y in _YAKUMAN
                                for y in yaku_names):
            from env.tenhou_eval import evaluate_hand
            player_wind = 27 + (seat - self.dealer) % 4
            round_wind = 27 + min(self.round_idx // 4, 2)
            ev = evaluate_hand(self.hands[seat], self.melds[seat], win_tile,
                               tsumo, player_wind, round_wind,
                               seat == self.dealer, result.yaku or [],
                               result.han,
                                   kiriage=self.cfg.kiriage_mangan)
            if ev is not None:
                han, fu = ev["han"], ev["fu"]
                yaku_names = list(ev["yaku"])
                from mahjong.hand_calculating.scores import ScoresCalculator
                cost = ScoresCalculator().calculate_scores(
                    han=han, fu=fu, config=config, is_yakuman=False)
        return {"han": han, "fu": fu, "yaku": yaku_names,
                "cost": cost, "level": cost["yaku_level"]}

    def _can_tsumo(self, seat: int) -> bool:
        if self.phase != "draw" or seat != self.turn or self.drawn_tile < 0:
            return False
        first_draw = not self._calls_made and self._draws_made[seat] == 1
        tenhou = first_draw and seat == self.dealer
        chiihou = first_draw and seat != self.dealer
        res = self._evaluate_win(
            seat, self.drawn_tile, tsumo=True,
            haitei=self._haitei_pending, rinshan=self._rinshan_current,
            tenhou=tenhou, chiihou=chiihou, daburu=self._daburu[seat])
        return res is not None

    def _can_ron(self, seat: int, tile: int, chankan: bool = False,
                 houtei: bool = False, kokushi_only: bool = False) -> bool:
        if seat == self._claim_discarder and self.phase == "claim":
            return False
        kind = kind_of(tile)
        if self.temp_furiten[seat] or self.riichi_furiten[seat]:
            return False
        wait = self._winning_kinds(seat, self.hands[seat])
        if kind not in wait:
            return False
        if wait & self._discard_kinds(seat):
            return False  # discard furiten
        res = self._evaluate_win(seat, tile, tsumo=False, chankan=chankan,
                                 houtei=houtei, daburu=self._daburu[seat])
        if res is None:
            return False
        if kokushi_only and not any("Kokushi" in y for y in res["yaku"]):
            return False  # robbing an ankan requires a kokushi win
        return True

    def _is_kokushi_13_wait(self, seat: int) -> bool:
        counts = counts34(self.hands[seat])
        for k in (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33):
            if counts[k] != 1:
                return False
        return True

    def _shanten_counts(self, seat: int, hand=None, exclude=None) -> list:
        """13/14-tile-equivalent 34-counts for the shanten checks: concealed +
        all meld tiles, dropping one tile per kan (the redundant 4th set
        member) so the mahjong lib always receives a valid hand size
        (multi-kan hands are physically 13+n_kan tiles, e.g. 15 for two kans)."""
        if hand is None:
            hand = self.hands[seat]
        counts = counts34([t for t in hand if t != exclude])
        for m in self.melds[seat]:
            for t in m["tiles"]:
                counts[t // 4] += 1
        n_kan = sum(1 for m in self.melds[seat]
                    if m["type"] in ("kan", "shouminkan"))
        for m in self.melds[seat]:
            if m["type"] in ("kan", "shouminkan") and n_kan > 0:
                counts[m["tiles"][0] // 4] -= 1
                n_kan -= 1
        return counts

    def _is_tenpai_shape(self, seat: int) -> bool:
        # Melds are FIXED sets: feeding their tiles to Shanten.calculate_shanten
        # lets the lib re-decompose them (e.g. a 567s meld + 4s + 77s pair can
        # become 456s + 777s and be wrongly called tenpai). Instead check the
        # win shape with melds fixed via Agari.is_agari(sets=...).
        hand = self.hands[seat]
        if not hand:
            return False
        if len(hand) == 14:
            # last drawer keeps 14 tiles: tenpai iff some 13-subset is tenpai
            return any(self._winning_kinds(seat, [t for t in hand if t != x])
                       for x in hand)
        return bool(self._winning_kinds(seat, hand))

    # ------------------------------------------------------ riichi / kyushu

    def _visible_counts(self) -> list:
        counts = [0] * 34
        for seat in range(4):
            for t in self.hands[seat]:
                counts[t // 4] += 1
            for m in self.melds[seat]:
                for t in m["tiles"]:
                    counts[t // 4] += 1
            for d in self.discards[seat]:
                counts[d["tile"] // 4] += 1
        for t in self.dora_indicators:
            counts[t // 4] += 1
        return counts

    def _riichi_options(self, seat: int) -> list:
        if self.riichi_declared[seat]:
            return []
        # ankan keeps menzen (门清): riichi after an ankan is legal; only
        # OPEN melds (chi/pon/shouminkan/daiminkan) block the declaration.
        if any(m["type"] in ("chi", "pon", "shouminkan") or m.get("open")
               for m in self.melds[seat]):
            return []
        if self.scores[seat] < 1000:
            return []
        if self.wall_left < 4:
            return []
        hand = self.hands[seat]
        # mjai.app parity: any tenpai-keeping discard may be riichi'd;
        # furiten or dead waits do not block the declaration. Ankans are
        # fixed sets: use the win-shape check (melds fixed) not the free
        # shanten decomposition.
        out = []
        for t in hand:
            hand13 = [x for x in hand if x != t]
            if self._winning_kinds(seat, hand13):
                out.append(t)
        return out

    def _kyushu_legal(self, seat: int) -> bool:
        if not self.cfg.kyushu_kyuhai:
            return False
        if self._draws_made[seat] != 1 or self._calls_made:
            return False
        kinds = {t // 4 for t in self.hands[seat]}
        terminal_honor = {k for k in kinds if is_terminal_or_honor(k)}
        return len(terminal_honor) >= 9

    # -------------------------------------------------------------- kan logic

    def _self_kan_options(self, seat: int) -> list:
        """Ankan + shouminkan option tile lists on the player's own turn."""
        hand = self.hands[seat]
        if self.n_kan >= 4 or self.wall_left < 1:
            return []
        opts = []
        counts = Counter(t // 4 for t in hand)
        drawn = self.drawn_tile
        for k, n in counts.items():
            if n >= 4:
                quad = sorted(t for t in hand if t // 4 == k)
                if not self.riichi_declared[seat] or (
                        drawn in quad and self._riichi_kan_ok(seat, k, drawn, True)):
                    opts.append(quad)
        for m in self.melds[seat]:
            if m["type"] != "pon":
                continue
            k = m["tiles"][0] // 4
            copies = [t for t in hand if t // 4 == k]
            for c in copies:
                if self.riichi_declared[seat]:
                    if c != drawn or not self._riichi_kan_ok(seat, k, drawn, False):
                        continue
                opts.append(sort_tiles136(list(m["tiles"]) + [c]))
        return opts

    def _riichi_kan_ok(self, seat: int, k: int, drawn: int, is_ankan: bool) -> bool:
        """After riichi a kan is legal only if the wait set is unchanged."""
        hand = self.hands[seat]
        pre13 = [t for t in hand if t != drawn]
        wait_pre = self._winning_kinds(seat, pre13)
        if not wait_pre:
            return False
        if not is_ankan:
            # shouminkan: the concealed 13 tiles are unchanged -> wait unchanged
            return True
        post13 = [t for t in pre13 if t // 4 != k]
        counts = counts34(post13)
        sets = self._meld_sets_for_agari(seat) + [[k] * 4]
        for ms in sets:
            for kk in ms:
                counts[kk] += 1
        wait_post = set()
        for kk in range(34):
            c = list(counts)
            c[kk] += 1
            if Agari.is_agari(c, sets or None):
                wait_post.add(kk)
        return wait_post == wait_pre

    def _daiminkan_options(self, seat: int, tile: int) -> list:
        if self.riichi_declared[seat] or self.n_kan >= 4:
            return []
        k = tile // 4
        copies = [t for t in self.hands[seat] if t // 4 == k]
        if len(copies) >= 3:
            return [sort_tiles136([tile] + copies)]
        return []

    def _pon_options(self, seat: int, tile: int) -> list:
        if self.riichi_declared[seat]:
            return []
        k = tile // 4
        copies = [t for t in self.hands[seat] if t // 4 == k]
        if len(copies) < 2:
            return []
        return [sort_tiles136([tile, a, b]) for a, b in itertools.combinations(copies, 2)]

    def _chow_options(self, seat: int, tile: int) -> list:
        if self.riichi_declared[seat]:
            return []
        kind = tile // 4
        if kind >= 27:
            return []
        base = kind // 9 * 9
        r = kind % 9
        hand = self.hands[seat]
        opts = []
        for start in (r - 2, r - 1, r):
            if start < 0 or start > 6:
                continue
            ks = [base + start, base + start + 1, base + start + 2]
            if kind not in ks:
                continue
            choices = []
            ok = True
            for k in ks:
                if k == kind:
                    choices.append([tile])
                else:
                    cps = [t for t in hand if t // 4 == k]
                    if not cps:
                        ok = False
                        break
                    choices.append(cps)
            if not ok:
                continue
            for combo in itertools.product(*choices):
                opts.append(sort_tiles136(combo))
        seen = set()
        uniq = []
        for o in opts:
            key = tuple(o)
            if key not in seen:
                seen.add(key)
                uniq.append(o)
        return uniq

    # ------------------------------------------------------------------- step

    def _normalize_action(self, action: dict, la: dict) -> dict:
        """Map MJAI-protocol action names onto the internal action types.

        Accepted aliases: dahai->discard, reach->riichi, chi->chow,
        hora->ron, ryuukyoku->kyushu, none->pass, and the split kan
        actions daiminkan/kakan/ankan. MJAI pai strings are resolved via
        parse_mjai_pai; when a pai alone does not identify the meld tiles
        unambiguously, the caller must pass explicit tiles.
        """
        a = dict(action)
        t = a["type"]
        if t == "dahai":
            a["type"] = "discard"
        elif t == "reach":
            a["type"] = "riichi"
        elif t == "chi":
            a["type"] = "chow"
        elif t == "hora":
            # MJAI hora with target==actor is a tsumo win
            if a.get("target") == a.get("actor"):
                a["type"] = "tsumo"
            else:
                a["type"] = "ron"
        elif t == "ryuukyoku":
            a["type"] = "kyushu"
        elif t == "none":
            a["type"] = "pass"
        elif t in ("daiminkan", "kakan", "ankan"):
            a["type"] = "kan"
            a["_kan_kind"] = t
        if "pai" in a and isinstance(a["pai"], str):
            tile = parse_mjai_pai(a["pai"])
            kind = tile // 4
            if a["type"] in ("discard", "riichi"):
                # pai strings carry no copy id: prefer the drawn tile, then
                # any legal hand copy of the same kind. The red/plain
                # identity of a five MUST be preserved: a plain "5m" string
                # must never be substituted by the red five (16) and vice
                # versa (verified against real Tenhou replay failures).
                candidates = la["discard"] + la["riichi"]
                if a.get("tsumogiri") and self.drawn_tile >= 0:
                    if self.drawn_tile // 4 == kind:
                        tile = self.drawn_tile
                elif tile not in candidates:
                    is_red = a["pai"].endswith("r")
                    chosen = None
                    for cand in candidates:
                        if cand // 4 == kind and (cand in RED_FIVES) == is_red:
                            chosen = cand
                            break
                    if chosen is None:
                        for cand in candidates:
                            if cand // 4 == kind:
                                chosen = cand
                                break
                    tile = chosen if chosen is not None else tile
            a["tile"] = tile
            a["pai"] = tile
        if a["type"] in ("chow", "pon", "kan") and "tiles" not in a:
            tile = a.get("tile")
            if tile is None:
                raise ValueError("chi/pon/kan need tiles or a pai")
            opts = la[a["type"]]
            matches = [o for o in opts
                       if any(t // 4 == tile // 4 for t in o["tiles"])]
            if len(matches) == 1:
                a["tiles"] = sorted(matches[0]["tiles"])
            elif len(matches) > 1:
                raise ValueError("pai alone is ambiguous; pass explicit tiles")
            else:
                raise ValueError("no legal meld uses tile %r" % (tile,))
        return a

    def step(self, action: dict) -> dict:
        """Apply one action for the current decision maker. Raises ValueError
        for illegal actions. Returns {"ok","events","round_end","game_end"}."""
        if self.phase in ("round_end", "game_end"):
            raise ValueError("no decisions in phase %s" % self.phase)
        if not isinstance(action, dict) or "type" not in action:
            raise ValueError("action must be a dict with a 'type'")
        seat = self.turn
        la = self.legal_actions(seat)
        action = self._normalize_action(action, la)
        atype = action["type"]
        if atype not in ("discard", "riichi", "chow", "pon", "kan", "ron", "tsumo", "kyushu", "pass"):
            raise ValueError("unknown action type %r" % atype)
        if atype in ("discard", "riichi") and action.get("tile", -1) not in la["discard" if atype == "discard" else "riichi"]:
            raise ValueError("illegal %s tile %r" % (atype, action.get("tile")))
        if atype in ("chow", "pon", "kan"):
            tiles = sorted(action.get("tiles", []))
            opts = la[atype]
            if atype == "kan" and self.phase == "claim":
                opts = la["kan"]
            if not any(sorted(o["tiles"]) == tiles for o in opts):
                raise ValueError("illegal %s tiles %r" % (atype, action.get("tiles")))
        if atype == "ron" and not la["ron"]:
            raise ValueError("illegal ron")
        if atype == "tsumo" and not la["tsumo"]:
            raise ValueError("illegal tsumo")
        if atype == "kyushu" and not la.get("kyushu"):
            raise ValueError("illegal kyushu")
        if atype == "pass" and self.phase not in ("claim", "chankan"):
            raise ValueError("illegal pass")

        if self.record_decisions and atype != "pass":
            rec = self.get_observation(seat)
            label = {"type": atype}
            if atype in ("discard", "riichi"):
                label["tile"] = action["tile"]
            if atype in ("chow", "pon", "kan"):
                label["tiles"] = sorted(action["tiles"])
            rec["label"] = label
            self.records.append(rec)

        n_events_before = len(self.events)
        if atype == "discard":
            self._do_discard(seat, action["tile"], with_riichi=False)
        elif atype == "riichi":
            self._do_discard(seat, action["tile"], with_riichi=True)
        elif atype == "tsumo":
            self._do_tsumo(seat)
        elif atype == "kyushu":
            self._ryuukyoku("kyushu")
        elif atype == "ron":
            self._do_ron(seat)
        elif atype == "pass":
            self._do_pass(seat)
        elif atype == "kan":
            self._do_kan_action(seat, sorted(action["tiles"]),
                                kan_kind=action.get("_kan_kind"))
        elif atype == "chow":
            self._do_claim(seat, "chow", sorted(action["tiles"]))
        elif atype == "pon":
            self._do_claim(seat, "pon", sorted(action["tiles"]))
        new_events = self.events[n_events_before:]
        ended = any(e["type"] in ("hora", "ryuukyoku") for e in new_events)
        return {
            "ok": True,
            "events": new_events,
            "round_end": self.round_result if ended else None,
            "game_end": self.game_result if self.phase == "game_end" else None,
        }

    # ------------------------------------------------------- discard / riichi

    def _do_discard(self, seat: int, tile: int, with_riichi: bool) -> None:
        if self._dora_at_discard is not None:
            self._flip_dora_indicator(self._dora_at_discard)
            self._dora_at_discard = None
        hand = self.hands[seat]
        if tile not in hand:
            raise ValueError("tile not in hand")
        hand.remove(tile)
        tsumogiri = tile == self.drawn_tile and self.drawn_tile >= 0
        if with_riichi:
            self._declare_riichi(seat)
        self.discards[seat].append({"tile": tile, "tsumogiri": tsumogiri, "riichi": with_riichi})
        if not self._calls_made:
            self._first_discards.append(tile // 4)
        self._last_discard_riichi = with_riichi
        self._houtei_pending = self.wall_left == 0
        self.last_event = {"type": "discard", "seat": seat, "tile": tile,
                           "riichi": with_riichi}
        self._emit({"type": "dahai", "actor": seat,
                    "pai": mjai_pai(tile),
                    "seat": seat, "tile": tile, "tsumogiri": tsumogiri})
        self.drawn_tile = -1
        self._start_claims(seat, tile)

    def _declare_riichi(self, seat: int) -> None:
        self.scores[seat] -= 1000
        self.kyoutaku += 1
        self.riichi_declared[seat] = True
        self._daburu[seat] = not self._calls_made and not self.discards[seat] and self.n_kan == 0
        self._riichi_draws[seat] = 0
        self._ippatsu_broken[seat] = False
        self._emit({"type": "reach", "actor": seat, "seat": seat})

    # ------------------------------------------------------------------ claims

    def _start_claims(self, disc_seat: int, tile: int) -> None:
        self.phase = "claim"
        self._claim_tile = tile
        self._claim_discarder = disc_seat
        self._ronners = []
        ron_cands = [s for s in self._seats_from(disc_seat)
                     if self._can_ron(s, tile, houtei=self._houtei_pending)]
        # Tenhou permits declining a ron to call instead (pon/chi/kan): the
        # discarder's tile can be claimed even when the hand could win on it.
        # ron keeps priority in the queue; after a pass the same seat may
        # still pon/chi/kan (oracle-verified: f0f2cf5f S3 S, seat 2 declined a
        # ron on 2s and pon'd it). Decline-ron sets same-go-around furiten,
        # which a successful call then lifts (_do_claim).
        kan_cands = [s for s in self._seats_from(disc_seat)
                     if self._daiminkan_options(s, tile)]
        pon_cands = [s for s in self._seats_from(disc_seat)
                     if self._pon_options(s, tile)]
        chi_seat = (disc_seat + 1) % 4
        chi_cands = [chi_seat] if self._chow_options(chi_seat, tile) else []
        q = [{"seat": s, "kind": "ron"} for s in ron_cands]
        q += [{"seat": s, "kind": "daiminkan"} for s in kan_cands]
        q += [{"seat": s, "kind": "pon"} for s in pon_cands]
        q += [{"seat": s, "kind": "chow"} for s in chi_cands]
        # same-go-around furiten tracking: every player whose hand *shape*
        # completes on this tile is recorded, so that a decline (pass or a
        # different call) sets temp-furiten until their next draw. Players
        # whose shape wins but who have NO yaku (and no call option) are NOT
        # queued at all: they have no choice to make ("無役は見逃しに
        # ならない"), they simply continue.
        self._shape_seers = [s for s in self._seats_from(disc_seat)
                             if tile // 4 in self._winning_kinds(s, self.hands[s])]
        self.claim_queue = q
        if not q:
            self._after_claims_pass()
            return
        self._set_turn_from_queue()

    def _set_turn_from_queue(self) -> None:
        if self.claim_queue:
            self.turn = self.claim_queue[0]["seat"]
        elif self.chankan_queue:
            self.turn = self.chankan_queue[0]["seat"]

    def _do_pass(self, seat: int) -> None:
        if self.phase == "claim":
            item = self.claim_queue.pop(0)
            if item["kind"] == "ron" or seat in self._shape_seers:
                self.temp_furiten[seat] = True
                if self.riichi_declared[seat]:
                    self.riichi_furiten[seat] = True
            if not self.claim_queue:
                if self._ronners:
                    self._settle_multi_ron()
                    return
                self._after_claims_pass()
            else:
                self._set_turn_from_queue()
        elif self.phase == "chankan":
            self.temp_furiten[seat] = True
            if self.riichi_declared[seat]:
                self.riichi_furiten[seat] = True
            self._ippatsu_broken[seat] = True  # declined the rob: ippatsu gone
            self.chankan_queue.pop(0)
            if not self.chankan_queue:
                if self._ronners:
                    self._settle_multi_ron()
                    return
                kan = self._kan_pending
                self._kan_pending = None
                self._finish_kan(kan["seat"], kan["kind"])
            else:
                self._set_turn_from_queue()

    def _after_claims_pass(self) -> None:
        if (self.cfg.four_winds_ryuukyoku and not self._calls_made
                and len(self._first_discards) == 4
                and all(k in self.WINDS_34 for k in self._first_discards)
                and len(set(self._first_discards)) == 1):
            self._ryuukyoku("four_winds")
            return
        if all(self.riichi_declared) and self._last_discard_riichi:
            self._ryuukyoku("four_riichi")
            return
        if self.wall_left <= 0:
            self._ryuukyoku("exhaustive")
            return
        self._enter_draw((self._claim_discarder + 1) % 4)

    def _do_claim(self, seat: int, kind: str, tiles: list) -> None:
        if kind not in ("chow", "pon"):
            raise ValueError("bad claim kind")
        # rules spec 2: a successful call lifts same-go-around furiten
        # immediately ("通过副露提前解除"); only passing keeps it set
        self.temp_furiten[seat] = False
        self.claim_queue = []
        self._calls_made = True
        for s in range(4):
            if s != seat:
                self._ippatsu_broken[s] = True
        tile = self._claim_tile
        from_seat = self._claim_discarder
        called = tile
        hand = self.hands[seat]
        for t in tiles:
            if t != tile:
                if t not in hand:
                    raise ValueError("claim tile not in hand")
                hand.remove(t)
        mtype = "chi" if kind == "chow" else "pon"
        meld = {"type": mtype, "tiles": tiles, "from": from_seat, "open": True,
                "called": called}
        self.melds[seat].append(meld)
        # 包赔（pao）：大三元/大四喜的第三/第四组役牌刻子由他人打出并鸣牌时，
        # 打出者对该玩家后续的和牌承担全部支付。
        if from_seat >= 0 and from_seat != seat:
            self._maybe_pao(seat, from_seat)
        self.discards[from_seat][-1]["called"] = True
        self._kuikae_kind = tile // 4
        self.last_event = {"type": kind, "seat": seat, "tile": tile}
        mjai_kind = "chi" if kind == "chow" else "pon"
        self._emit({"type": mjai_kind, "actor": seat, "target": from_seat,
                    "pai": mjai_pai(tile),
                    "consumed": [mjai_pai(t) for t in tiles if t != tile],
                    "seat": seat, "tile": tile, "from": from_seat,
                    "tiles": list(tiles)})
        self.drawn_tile = -1
        self.turn = seat
        self.phase = "draw"

    def _maybe_pao(self, seat: int, from_seat: int) -> None:
        """包赔判定：鸣牌后若形成大三元(3组三元)或大四喜(4组风)役牌刻子，
        且本组来自他人打出，则 from_seat 包赔（承担该玩家和牌的全部支付）。"""
        import os as _os
        if _os.environ.get("PAO_DEBUG"):
            print("PAO-IN: seat=%d from=%d melds=%s" % (seat, from_seat,
                  [[m["type"], m["tiles"][0]//4] for m in self.melds[seat]]), flush=True)
        honor = [m for m in self.melds[seat]
                 if m["type"] in ("pon", "kan", "shouminkan")
                 and m["tiles"][0] // 4 >= 27]
        san_gen = {m["tiles"][0] // 4 for m in honor
                   if m["tiles"][0] // 4 in (31, 32, 33)}
        winds = {m["tiles"][0] // 4 for m in honor
                 if m["tiles"][0] // 4 in (27, 28, 29, 30)}
        if len(san_gen) >= 3 or len(winds) >= 4:
            self._pao[seat] = from_seat

    def _do_kan_action(self, seat: int, tiles: list, kan_kind=None) -> None:
        if self.phase == "draw":
            hand = self.hands[seat]
            counts = Counter(t // 4 for t in hand)
            if kan_kind in (None, "ankan"):
                if any(counts[k] >= 4 and sorted(t for t in hand if t // 4 == k) == tiles
                       for k in counts):
                    self._execute_kan(seat, "ankan", tiles, from_seat=-1, called=-1)
                    return
                if kan_kind == "ankan":
                    raise ValueError("illegal ankan tiles")
            if kan_kind in (None, "kakan"):
                for m in self.melds[seat]:
                    if m["type"] == "pon":
                        for c in self.hands[seat]:
                            if (c // 4 == m["tiles"][0] // 4
                                    and sort_tiles136(list(m["tiles"]) + [c]) == tiles):
                                self._execute_kan(seat, "shouminkan", tiles,
                                                  from_seat=m["from"], called=c)
                                return
                if kan_kind == "kakan":
                    raise ValueError("illegal kakan tiles")
            raise ValueError("illegal kan tiles")
        elif self.phase == "claim":
            if kan_kind not in (None, "daiminkan"):
                raise ValueError("only daiminkan is allowed on a claim")
            # rules spec 2: a successful kan call lifts same-go-around
            # furiten immediately
            self.temp_furiten[seat] = False
            self.claim_queue = []
            self._execute_kan(seat, "daiminkan", tiles,
                              from_seat=self._claim_discarder, called=self._claim_tile)
        else:
            raise ValueError("kan not allowed in this phase")

    def _execute_kan(self, seat: int, kind: str, tiles: list, from_seat: int,
                     called: int) -> None:
        if kind == "ankan" and self._dora_at_discard is not None:
            self._flip_dora_indicator(self._dora_at_discard)
            self._dora_at_discard = None
        hand = self.hands[seat]
        if kind == "ankan":
            for t in tiles:
                if t not in hand:
                    raise ValueError("ankan tiles not in hand")
                hand.remove(t)
            meld = {"type": "kan", "tiles": tiles, "from": from_seat, "open": False,
                    "called": -1}
            self.melds[seat].append(meld)
        elif kind == "daiminkan":
            for t in tiles:
                if t != called and t not in hand:
                    raise ValueError("daiminkan tiles not in hand")
            for t in tiles:
                if t != called:
                    hand.remove(t)
            meld = {"type": "kan", "tiles": tiles, "from": from_seat, "open": True,
                    "called": called}
            self.melds[seat].append(meld)
            self.discards[from_seat][-1]["called"] = True
        elif kind == "shouminkan":
            k = tiles[0] // 4
            if called not in hand:
                raise ValueError("shouminkan tile not in hand")
            hand.remove(called)
            for i, m in enumerate(self.melds[seat]):
                if m["type"] == "pon" and m["tiles"][0] // 4 == k:
                    self.melds[seat][i] = {"type": "shouminkan", "tiles": tiles,
                                           "from": m["from"], "open": True,
                                           "called": called}
                    break
            else:
                raise ValueError("no matching pon for shouminkan")
        else:
            raise ValueError("bad kan kind")
        self.n_kan += 1
        self.kan_owners.append(seat)
        self._calls_made = True
        self._kuikae_kind = tiles[0] // 4
        self.last_event = {"type": "kan", "seat": seat,
                           "tile": called if called >= 0 else tiles[0]}
        mjai_kind = {"ankan": "ankan", "daiminkan": "daiminkan",
                     "shouminkan": "kakan"}[kind]
        called_tile = called if called >= 0 else tiles[0]
        consumed = ([mjai_pai(t) for t in tiles] if kind == "ankan"
                    else [mjai_pai(t) for t in tiles if t != called_tile])
        self._emit({"type": mjai_kind, "actor": seat,
                    "pai": mjai_pai(called_tile),
                    "consumed": consumed,
                    "seat": seat, "tiles": list(tiles), "from": from_seat})
        # chankan window (rob the kan): only a kakan (shouminkan) can be
        # robbed, by any winning hand; an ankan can never be robbed
        # (platform v1.7 / Tenhou).
        cands = []
        if kind == "shouminkan":
            cands = [s for s in self._seats_from(seat)
                     if self._can_ron(s, called, chankan=True)]
        # ippatsu: ankan/daiminkan break everyone else's ippatsu; a kakan
        # keeps the would-be robbers' ippatsu alive until the chankan window
        # closes (mjai.app parity: robbing a kakan preserves ippatsu).
        if kind == "shouminkan":
            for s in range(4):
                if s != seat and s not in cands:
                    self._ippatsu_broken[s] = True
        else:
            for s in range(4):
                if s != seat:
                    self._ippatsu_broken[s] = True
        if cands:
            self._kan_pending = {"seat": seat,
                                 "tile": called if called >= 0 else tiles[0],
                                 "kind": kind}
            self.chankan_queue = [{"seat": s} for s in cands]
            self._ronners = []
            self.phase = "chankan"
            self._set_turn_from_queue()
        else:
            self._finish_kan(seat, kind)

    def _flip_dora_indicator(self, pos: int) -> None:
        self.dora_indicators.append(self.wall[pos])
        self._emit({"type": "dora", "dora_marker": mjai_pai(self.dora_indicators[-1]),
                    "tile": self.dora_indicators[-1]})

    def _finish_kan(self, seat: int, kind: str = "ankan") -> None:
        # mjai.app parity: ankan reveals its dora immediately; daiminkan/kakan
        # reveal it at the kan players next discard (or next tsumo if they
        # kakan again before discarding).
        if kind == "ankan":
            self._flip_dora_indicator(126 + self.n_kan)
        else:
            if self._dora_at_discard is not None:
                self._dora_at_tsumo = self._dora_at_discard
            self._dora_at_discard = 126 + self.n_kan
        if (self.n_kan == 4 and self.cfg.four_kans_ryuukyoku
                and len(set(self.kan_owners)) > 1):
            self._ryuukyoku("four_kans")
            return
        # mjai.app parity: the rinshan replacement counts as one of the
        # 70 draws (no wall_left increment); with K kans only 70-K
        # tiles are ever drawn from the live wall.
        self._rinshan_pending = True
        self._enter_draw(seat)

    # --------------------------------------------------------------- ron/tsumo

    def _do_ron(self, seat: int) -> None:
        if self.phase == "claim":
            if self.cfg.atamahane and not self.cfg.double_ron:
                self.claim_queue = []
                tile = self._claim_tile
                self._settle_single_ron(seat, tile, chankan=False)
                return
            self._ronners.append(seat)
            if len(self._ronners) >= 3:
                # 三家和: abortive draw; no settlement, the dealer renchans
                # and the riichi sticks stay in the pool
                self._ryuukyoku("ron3")
                return
            self.claim_queue = [it for it in self.claim_queue[1:]
                                if it["kind"] == "ron"]
            if not self.claim_queue:
                self._settle_multi_ron()
            else:
                self._set_turn_from_queue()
        elif self.phase == "chankan":
            if self.cfg.atamahane and not self.cfg.double_ron:
                self.chankan_queue = []
                kan = self._kan_pending
                self._settle_single_ron(seat, kan["tile"], chankan=True)
                return
            self._ronners.append(seat)
            self.chankan_queue = self.chankan_queue[1:]
            if not self.chankan_queue:
                self._settle_multi_ron()
            else:
                self._set_turn_from_queue()
        else:
            raise ValueError("ron not allowed in this phase")

    def _do_tsumo(self, seat: int) -> None:
        self.phase = "round_end"
        self._settle_tsumo(seat)

    # ------------------------------------------------------------- settlement

    def _win_record(self, seat: int, res: dict) -> dict:
        return {"seat": seat, "han": res["han"], "fu": res["fu"],
                "yaku": res["yaku"], "level": res["level"],
                "is_dealer": seat == self.dealer}

    def _settle_single_ron(self, seat: int, tile: int, chankan: bool) -> None:
        if chankan:
            disc_seat = self._kan_pending["seat"]
            self._kan_pending = None
        else:
            disc_seat = self._claim_discarder
        res = self._evaluate_win(seat, tile, tsumo=False, chankan=chankan,
                                 houtei=self._houtei_pending,
                                 daburu=self._daburu[seat])
        if res is None:
            raise ValueError("hand cannot win")
        self._apply_ron_payments([(seat, res)], disc_seat)
        self.kyoutaku = 0
        self.round_result = {
            "type": "ron", "winners": [self._win_record(seat, res)],
            "loser": disc_seat, "tile": tile, "chankan": chankan,
            "scores": list(self.scores), "honba": self.honba,
            "round": self.round_idx, "riichi_sticks": self.kyoutaku,
        }
        self.phase = "round_end"
        self._emit({"type": "hora", "actor": seat, "target": disc_seat,
                    "pai": mjai_pai(tile), "seat": seat, "tile": tile,
                    "han": res["han"], "fu": res["fu"]})
        self._emit({"type": "end_kyoku"})
        self._advance_round(renchan=(seat == self.dealer),
                            dealer_win=(seat == self.dealer))

    def _settle_multi_ron(self) -> None:
        if not self._ronners:
            raise ValueError("no ronners")
        tile = self._claim_tile if self.phase == "claim" else self._kan_pending["tile"]
        chankan = self.phase == "chankan"
        disc_seat = self._claim_discarder if not chankan else self._kan_pending["seat"]
        if chankan:
            self._kan_pending = None
        results = []
        for s in self._ronners:
            res = self._evaluate_win(s, tile, tsumo=False, chankan=chankan,
                                     houtei=self._houtei_pending,
                                     daburu=self._daburu[s])
            if res is not None:
                results.append((s, res))
        self._apply_ron_payments(results, disc_seat)
        self.kyoutaku = 0
        self.round_result = {
            "type": "ron", "winners": [self._win_record(s, r) for s, r in results],
            "loser": disc_seat, "tile": tile, "chankan": chankan,
            "multiple": True, "scores": list(self.scores), "honba": self.honba,
            "round": self.round_idx, "riichi_sticks": self.kyoutaku,
        }
        self.phase = "round_end"
        dealer_won = self.dealer in [s for s, _ in results]
        for s, r in results:
            self._emit({"type": "hora", "actor": s, "target": disc_seat,
                        "pai": mjai_pai(tile), "seat": s, "tile": tile,
                        "han": r["han"], "fu": r["fu"]})
        self._emit({"type": "end_kyoku"})
        self._advance_round(renchan=dealer_won, dealer_win=dealer_won)

    def _apply_ron_payments(self, winners: list, disc_seat: int) -> None:
        # mjai.app/Tenhou double-ron parity: the ronner closest to the
        # discarder takes the full riichi-stick pool and all honba; the
        # others collect only the base payment.
        ordered = sorted(winners, key=lambda pair: (pair[0] - disc_seat) % 4)
        first = True
        for s, res in ordered:
            c = res["cost"]
            pay = c["main"] + (c["main_bonus"] if first else 0)
            payer = self._pao.get(s)
            if payer is not None and payer != disc_seat:
                # 天凤包赔荣和：包赔者与放铳者各付一半点数（向上取整到
                # 百位后平分），本场由包赔者承担；供托归和牌者
                # （oracle-verified: 83b92b4d E2, 大三元荣和 32000：
                # 包赔者 16600(=16000+本场600)、放铳者 16000）
                base = c["main"]
                half = (base // 2 + 99) // 100 * 100 if base % 200 else base // 2
                pao_pay = half + (c["main_bonus"] if first else 0)
                disc_pay = base - half
                self.scores[s] += pay + (self.kyoutaku if first else 0) * 1000
                self.scores[payer] -= pao_pay
                self.scores[disc_seat] -= disc_pay
            else:
                self.scores[s] += pay + (self.kyoutaku if first else 0) * 1000
                self.scores[payer if payer is not None else disc_seat] -= pay
            first = False

    def _settle_tsumo(self, seat: int) -> None:
        first_draw = not self._calls_made and self._draws_made[seat] == 1
        res = self._evaluate_win(
            seat, self.drawn_tile, tsumo=True,
            haitei=self._haitei_pending, rinshan=self._rinshan_current,
            tenhou=first_draw and seat == self.dealer,
            chiihou=first_draw and seat != self.dealer,
            daburu=self._daburu[seat])
        if res is None:
            raise ValueError("hand cannot win")
        c = res["cost"]
        gross = c["main"] + c["main_bonus"] + 2 * (c["additional"] + c["additional_bonus"])
        pao = self._pao.get(seat)
        if pao is not None:
            # 包赔（大三元/大四喜/四杠子）：包赔者单付全部，其余不付
            self.scores[pao] -= gross
            self.scores[seat] += gross + self.kyoutaku * 1000
            self.kyoutaku = 0
        else:
            self.scores[seat] += gross + self.kyoutaku * 1000
            self.kyoutaku = 0
            if seat == self.dealer:
                for s in range(4):
                    if s != seat:
                        self.scores[s] -= c["main"] + c["main_bonus"]
            else:
                self.scores[self.dealer] -= c["main"] + c["main_bonus"]
                for s in range(4):
                    if s != seat and s != self.dealer:
                        self.scores[s] -= c["additional"] + c["additional_bonus"]
        self.round_result = {
            "type": "tsumo", "winners": [self._win_record(seat, res)],
            "tile": self.drawn_tile, "scores": list(self.scores),
            "honba": self.honba, "round": self.round_idx,
            "riichi_sticks": self.kyoutaku,
        }
        self.phase = "round_end"
        self._emit({"type": "hora", "actor": seat,
                    "pai": mjai_pai(self.drawn_tile), "seat": seat,
                    "tile": self.drawn_tile, "han": res["han"], "fu": res["fu"]})
        self._emit({"type": "end_kyoku"})
        self._advance_round(renchan=(seat == self.dealer),
                            dealer_win=(seat == self.dealer))

    # --------------------------------------------------------------- ryuukyoku

    def _is_nagashi(self, seat: int) -> bool:
        ds = self.discards[seat]
        if not ds:
            return False
        for d in ds:
            if not is_terminal_or_honor(d["tile"] // 4):
                return False
            if d.get("called"):
                return False
        return True

    def _ryuukyoku(self, rtype: str) -> None:
        self.phase = "round_end"
        tenpai = [s for s in range(4) if self._is_tenpai_shape(s)]
        nagashi = []
        renchan = None
        if rtype == "exhaustive":
            if self.cfg.nagashi_mangan:
                nagashi = [s for s in range(4) if self._is_nagashi(s)]
            self._settle_ryuukyoku_points(tenpai, nagashi)
            renchan = self.dealer in tenpai or self.dealer in nagashi
        else:
            renchan = True  # kyushu / four_winds / four_kans / four_riichi
        # Tenhou/standard rule (verified against 77/77 real-log cases): riichi
        # sticks standing on the table are RETURNED to the players who
        # declared riichi when the round ends in a draw (exhaustive or
        # abortive). Game-end leftovers still go to the top player via
        # _end_game.
        if self.kyoutaku > 0:
            for s in range(4):
                if self.riichi_declared[s] and self.kyoutaku > 0:
                    self.scores[s] += 1000
                    self.kyoutaku -= 1
            self.kyoutaku = 0
        self.round_result = {
            "type": "ryuukyoku:" + rtype, "tenpai": tenpai, "nagashi": nagashi,
            "scores": list(self.scores), "honba": self.honba,
            "round": self.round_idx, "riichi_sticks": self.kyoutaku,
            "renchan": renchan,
        }
        self._emit({"type": "ryuukyoku", "reason": rtype,
                    "tenpais": [str(s) for s in tenpai],
                    "tenpai": list(tenpai), "nagashi": list(nagashi)})
        self._emit({"type": "end_kyoku"})
        self._advance_round(renchan, dealer_win=False)

    def _settle_ryuukyoku_points(self, tenpai: list, nagashi: list) -> None:
        # nagashi mangan: tsumo-mangan settlement per winner
        for s in nagashi:
            if s == self.dealer:
                pay = 4000
                for o in range(4):
                    if o != s:
                        self.scores[o] -= pay
                        self.scores[s] += pay
            else:
                self.scores[self.dealer] -= 4000
                self.scores[s] += 4000
                for o in range(4):
                    if o != s and o != self.dealer:
                        self.scores[o] -= 2000
                        self.scores[s] += 2000
        # Tenhou: 流局满贯 REPLACES the tenpai-noten settlement entirely
        # ("流局罚符以満贯清算代替"); skip the 3000-point settlement whenever
        # a nagashi mangan winner exists.
        if nagashi:
            return
        # 3000-point tenpai settlement among the rest
        tp = [s for s in tenpai if s not in nagashi]
        noten = [s for s in range(4) if s not in tenpai and s not in nagashi]
        if len(tp) == 1:
            self.scores[tp[0]] += 3000
            for s in noten:
                self.scores[s] -= 1000
        elif len(tp) == 2:
            for s in tp:
                self.scores[s] += 1500
            for s in noten:
                self.scores[s] -= 1500
        elif len(tp) == 3:
            for s in tp:
                self.scores[s] += 1000
            for s in noten:
                self.scores[s] -= 3000

    # ------------------------------------------------------- round advancement

    def _advance_round(self, renchan: bool, dealer_win: bool = False) -> None:
        if not renchan:
            self.dealer = (self.dealer + 1) % 4
            self.round_idx += 1
            self.honba = 0
        else:
            self.honba += 1
        # tobi: any negative score ends the game immediately
        if min(self.scores) < 0:
            self._end_game()
            return
        # agari-yame: dealer win at all-last ends the game only when the
        # dealer has return_score+ and is (joint) top.
        if (renchan and dealer_win and self.round_idx >= 7 and self.cfg.agari_yame
                and self.scores[self.dealer] >= self.cfg.return_score
                and self.scores[self.dealer] == max(self.scores)):
            self._end_game()
            return
        if self.round_idx >= 8:
            if not self.cfg.west_extension:
                self._end_game()
                return
            # West-round continuation (Tenhou/Majsoul style): keep playing
            # only while every score is still below return_score (negative
            # scores were handled by tobi above); hard cap after West-4.
            if self.round_idx >= 12 or max(self.scores) >= self.cfg.return_score:
                self._end_game()
                return
        self._new_round()

    def _end_game(self) -> None:
        if self.kyoutaku > 0:
            # leftover riichi sticks go to the (joint) top player
            self.scores[max(range(4), key=lambda s: self.scores[s])] += self.kyoutaku * 1000
            self.kyoutaku = 0
        order = sorted(range(4), key=lambda s: -self.scores[s])
        self.game_result = {
            "scores": list(self.scores),
            "rank": [s + 1 for s in order],
            "order": order,
            "rounds_played": self._game_no,
        }
        self.phase = "game_end"
        self._emit({"type": "end_game", "scores": list(self.scores)})

    # ------------------------------------------------------------- observation

    def get_observation(self, seat=None, include_explain: bool = False) -> dict:
        """Schema §2 observation dict. `seat` selects the hand to reveal;
        legal_actions are filled only for the current decision maker."""
        if seat is None:
            seat = self.turn
        decider = self.turn
        obs = {
            "game_id": self.game_id,
            "source": "engine",
            "seat": seat,
            "round": self.round_idx if self.round_idx <= 7 else -1,
            "honba": self.honba,
            "riichi_sticks": self.kyoutaku,
            "wall_left": self.wall_left,
            "dora_indicators": list(self.dora_indicators),
            "scores": list(self.scores),
            "oya": self.dealer,
            "hand": sort_tiles136(list(self.hands[seat])),
            "melds": [[self._meld_view(m) for m in self.melds[s]] for s in range(4)],
            "discards": [[dict(d) for d in self.discards[s]] for s in range(4)],
            "n_kan": self.n_kan,
            "riichi_declared": list(self.riichi_declared),
            "last_event": dict(self.last_event),
            "legal_actions": self.legal_actions(seat) if seat == decider else _empty_legal(),
            "label": None,
        }
        if include_explain:
            from riichi.explain import explain_decision
            obs["explain"] = explain_decision(self, seat)
        return obs

    def _meld_view(self, m: dict) -> dict:
        mtype = "kan" if m["type"] in ("kan", "shouminkan") else m["type"]
        return {"type": mtype, "tiles": list(m["tiles"]),
                "from": m["from"], "red": any(t in RED_FIVES for t in m["tiles"])}

    # ------------------------------------------------------------ convenience

    def random_action(self, rng=None) -> dict:
        """A simple random policy over the current legal actions."""
        if rng is None:
            rng = random.Random(0)
        seat = self.turn
        la = self.legal_actions(seat)
        if self.phase in ("claim", "chankan"):
            if la["ron"] and rng.random() < 0.9:
                return {"type": "ron"}
            if rng.random() < 0.5:
                return {"type": "pass"}
            if la["kan"]:
                return {"type": "kan", "tiles": list(rng.choice(la["kan"])["tiles"])}
            if la["pon"]:
                return {"type": "pon", "tiles": list(rng.choice(la["pon"])["tiles"])}
            if la["chow"]:
                return {"type": "chow", "tiles": list(rng.choice(la["chow"])["tiles"])}
            return {"type": "pass"}
        if la["tsumo"] and rng.random() < 0.95:
            return {"type": "tsumo"}
        if la.get("kyushu") and rng.random() < 0.8:
            return {"type": "kyushu"}
        if la["riichi"] and rng.random() < 0.3:
            return {"type": "riichi", "tile": rng.choice(la["riichi"])}
        if la["kan"] and rng.random() < 0.3:
            return {"type": "kan", "tiles": list(rng.choice(la["kan"])["tiles"])}
        return {"type": "discard", "tile": rng.choice(la["discard"])}

    def play_random(self, max_steps: int = 10000, seed: int = 1) -> dict:
        """Play the game to the end with the random policy."""
        rng = random.Random(seed)
        for _ in range(max_steps):
            if self.phase == "game_end":
                break
            self.step(self.random_action(rng))
        else:
            raise RuntimeError("game did not finish in %d steps" % max_steps)
        return self.game_result

    def debug_setup(self, hands=None, dealer: int = 0, wall=None, discards=None,
                    melds=None, scores=None, honba: int = 0, kyoutaku: int = 0,
                    riichi=None, n_kan: int = 0, round_idx: int = 0,
                    draw_ptr: int = 0, wall_left: int = 70,
                    dora_indicators=None, phase: str = "draw") -> None:
        """Deterministic test setup. `wall` must be a full permutation of 0..135.
        hands: 13 tiles per seat; the dealer then draws normally."""
        assert len(wall) == 136 and sorted(wall) == list(range(136)), "wall must be 0..135"
        self.wall = list(wall)
        self.draw_ptr = draw_ptr
        self.rinshan_ptr = 125
        self.wall_left = wall_left
        self.hands = [sort_tiles136(list(h)) for h in hands]
        self.dealer = dealer
        self.round_idx = round_idx
        self.honba = honba
        self.kyoutaku = kyoutaku
        self.scores = list(scores) if scores else [25000] * 4
        self.melds = [list(x) for x in melds] if melds else [[] for _ in range(4)]
        self.discards = [[dict(d) for d in x] for x in discards] if discards else [[] for _ in range(4)]
        self.riichi_declared = list(riichi) if riichi else [False] * 4
        self.riichi_furiten = [False] * 4
        self.temp_furiten = [False] * 4
        self._riichi_draws = [0] * 4
        self._ippatsu_broken = [False] * 4
        self._daburu = [False] * 4
        self._has_drawn = [bool(self.discards[s]) for s in range(4)]
        self._has_drawn[dealer] = True
        self._draws_made = [0] * 4
        self._calls_made = any(self.melds)
        self._first_discards = []
        self._draw_counter = 0
        self.n_kan = n_kan
        self.kan_owners = []
        self.dora_indicators = list(dora_indicators) if dora_indicators else [self.wall[126]]
        self.phase = phase
        self.claim_queue = []
        self.chankan_queue = []
        self._shape_seers = []
        self._ronners = []
        self._pao = {}
        self._kan_pending = None
        self.drawn_tile = -1
        self._rinshan_pending = False
        self._dora_at_tsumo = None
        self._dora_at_discard = None
        self._haitei_pending = False
        self._houtei_pending = False
        self._kuikae_kind = -1
        self._last_discard_riichi = False
        self.round_result = None
        self.game_result = None
        self.last_event = {"type": "debug", "seat": -1, "tile": -1}
        self.events = []  # drop the construction-time random round events
        if phase == "draw":
            self._emit_start_kyoku()
            self._enter_draw(dealer)

    def __repr__(self) -> str:
        return ("<RiichiGame round=%d honba=%d phase=%s turn=%d wall_left=%d "
                "scores=%s>" % (self.round_idx, self.honba, self.phase, self.turn,
                                self.wall_left, self.scores))
