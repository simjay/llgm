"""Frozen, independently themed development histories for real retrieval diagnostics.

These are controlled synthetic records, not a benchmark or sampled conversations.
Source text is generated deterministically from authored notes before retrieval is
run. Evaluator annotations are returned separately and never enter source metadata.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llgm.core.types import SourceNode, Turn
from llgm.evaluation.longmemeval import EvaluationCase, GoldRecord

SEED = 20260911
SOURCE_COUNT = 96
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "experiments/node_search_controlled.json"

# Each history uses different people, places, activities, and inventory vocabulary.
_THEMES = (
    (
        "Bellhaven Museum",
        ("Mira", "Jonas", "Adele", "Ravi", "Esme", "Noel"),
        ("print room", "sculpture hall", "archive", "courtyard", "studio", "west gallery"),
        (
            "frame inspection",
            "catalog photography",
            "school visit",
            "lighting review",
            "label proofing",
            "loan packing",
        ),
        (
            "hanging wires",
            "cotton gloves",
            "mounting boards",
            "display plinths",
            "photo sleeves",
            "storage trays",
        ),
    ),
    (
        "Cinder Cafe",
        ("Petra", "Owen", "Luca", "Tessa", "Hugh", "Nadia"),
        ("roasting bench", "terrace", "service counter", "pantry", "wash station", "back office"),
        (
            "bean delivery",
            "pastry tasting",
            "grinder calibration",
            "menu lettering",
            "terrace cleaning",
            "till reconciliation",
        ),
        (
            "coffee filters",
            "paper cups",
            "bread baskets",
            "aprons",
            "stirring spoons",
            "receipt rolls",
        ),
    ),
    (
        "Arden Depot",
        ("Soren", "Leila", "Bram", "Inez", "Felix", "Dora"),
        (
            "dispatch desk",
            "north bay",
            "loading yard",
            "packing aisle",
            "returns cage",
            "inspection lane",
        ),
        (
            "pallet counting",
            "scanner training",
            "vehicle inspection",
            "label printing",
            "crate washing",
            "manifest reconciliation",
        ),
        (
            "shipping sleeves",
            "ratchet straps",
            "plastic crates",
            "barcode stickers",
            "packing tape",
            "wheel chocks",
        ),
    ),
    (
        "Cedar Campus",
        ("Anika", "Caleb", "Greta", "Elias", "Sofia", "Duncan"),
        (
            "student commons",
            "sports pavilion",
            "east gate",
            "workshop block",
            "residence office",
            "garden court",
        ),
        (
            "noticeboard revision",
            "bicycle inspection",
            "sports booking",
            "locker inventory",
            "garden maintenance",
            "welcome planning",
        ),
        (
            "visitor lanyards",
            "cycle pumps",
            "folding tables",
            "sports bibs",
            "desk lamps",
            "garden gloves",
        ),
    ),
    (
        "Estuary Aquarium",
        ("Hana", "Victor", "Mabel", "Idris", "Clara", "Wesley"),
        (
            "kelp exhibit",
            "touch pool",
            "pump room",
            "education lab",
            "quarantine suite",
            "ticket booth",
        ),
        (
            "salinity sampling",
            "feeding observation",
            "glass cleaning",
            "filter maintenance",
            "lesson preparation",
            "habitat photography",
        ),
        (
            "sample bottles",
            "feeding tongs",
            "cleaning pads",
            "hose fittings",
            "laminated diagrams",
            "specimen trays",
        ),
    ),
    (
        "Larch Sound",
        ("Nell", "Ronan", "Yara", "Stefan", "June", "Malik"),
        (
            "rehearsal tent",
            "ticket cabin",
            "river stage",
            "costume store",
            "media room",
            "volunteer lounge",
        ),
        (
            "cable inspection",
            "poster distribution",
            "ticket accounting",
            "costume fitting",
            "stage cleaning",
            "microphone testing",
        ),
        (
            "audio cables",
            "poster tubes",
            "wristbands",
            "costume hangers",
            "music stands",
            "battery packs",
        ),
    ),
    (
        "Northline Bistro",
        ("Imogen", "Theo", "Amara", "Cedric", "Lena", "Oscar"),
        (
            "prep kitchen",
            "courtyard tables",
            "dry store",
            "dish room",
            "bar station",
            "booking desk",
        ),
        (
            "knife sharpening",
            "linen inventory",
            "menu printing",
            "extractor cleaning",
            "glass polishing",
            "reservation review",
        ),
        (
            "tea towels",
            "serving boards",
            "table candles",
            "mixing bowls",
            "water jugs",
            "order pads",
        ),
    ),
    (
        "Quarry Library",
        ("Beatrice", "Kian", "Alma", "Francis", "Zuri", "Edwin"),
        (
            "reading room",
            "returns desk",
            "children's corner",
            "local history room",
            "binding bench",
            "garden annexe",
        ),
        (
            "shelf checking",
            "book repair",
            "reading club planning",
            "catalog correction",
            "display assembly",
            "donation sorting",
        ),
        (
            "book supports",
            "repair tape",
            "index cards",
            "display boards",
            "dust jackets",
            "archive folders",
        ),
    ),
)


def _node(case_id: str, ordinal: int, texts: list[str]) -> SourceNode:
    """Assign opaque stable source identifiers without exposing record roles."""
    identity = hashlib.sha256(f"{SEED}:{case_id}:{ordinal}".encode()).hexdigest()[:20]
    return SourceNode(
        node_id=f"n-{identity}",
        turns=tuple(Turn(f"t{index:05d}", "user", text) for index, text in enumerate(texts)),
        metadata={},
    )


def _background(theme_index: int, ordinal: int, rng: random.Random) -> list[str]:
    """Write a routine operational record using independently varied details."""
    organization, people, places, activities, supplies = _THEMES[theme_index]
    person, colleague = rng.sample(people, 2)
    place, second_place = rng.sample(places, 2)
    activity, followup = rng.sample(activities, 2)
    supply, spare = rng.sample(supplies, 2)
    day = 1 + ordinal % 27
    quantity = rng.randrange(8, 76)
    shelf = rng.choice(("upper", "middle", "lower"))
    weather = rng.choice(("light rain", "clear skies", "strong wind", "a humid afternoon"))
    meeting = rng.choice(("Monday", "Wednesday", "Friday"))
    observation = rng.choice(
        (
            "The handwritten count differed from the spreadsheet by two, so we counted the sealed boxes again before signing.",
            "One carton arrived damp, but the wrapping inside was intact and none of the contents needed replacing.",
            "The delivery driver called ahead, which gave us enough time to clear the passage and find an empty trolley.",
            "The previous sheet used an old product name; we checked the supplier description instead of ordering a duplicate.",
            "A temporary label had fallen off overnight, so we compared the remaining packages with the purchase receipt.",
            "The afternoon team found the transfer note beside the register and added the missing initials before closing.",
        )
    )
    return [
        (
            f"August {day}, work notebook at {organization}. {person} and I reviewed {activity} in the {place}. "
            f"There were {quantity} usable {supply}; the spare {spare} were on the {shelf} shelf. {observation} "
            f"We placed the signed sheet in the office folder and asked {colleague} to bring it to the {meeting} meeting. "
            f"The plan is to finish the physical check before changing any order quantities. This visit concerned routine "
            f"stock and housekeeping, and the receipts remain with this notebook entry."
        ),
        (
            f"During {weather} at {organization}, {colleague} met me in the {second_place} to discuss {followup}. "
            f"We agreed to work in two short sessions so the {place} can stay usable. {person} will photograph the "
            f"arrangement before anything is moved, and I will put the {spare} into separate marked boxes. The team "
            f"asked for a sketch showing the clear walking route between the storage shelves. We also need to check "
            f"whether the old trolley wheels leave marks on the floor. I recorded this as notebook item {ordinal + 101}, "
            f"with the supplier receipt attached to the same page."
        ),
        (
            f"Follow-up for notebook item {ordinal + 101} at {organization}: the {activity} was finished and the "
            f"{supply} count was checked against the earlier list. {colleague} noticed that one cupboard door sticks "
            f"after wet weather, so facilities will look at the hinge at its next visit. We kept the contents on the "
            f"{shelf} shelf until then. {person} asked everyone to return borrowed equipment to the {second_place} "
            f"and write down anything used up. The completed form belongs with the August receipts, while a clean "
            f"copy of the checklist will be printed for the next {followup}."
        ),
    ]


def _crowding_notes(
    organization: str, topic: str, subject: str, details: tuple[str, ...]
) -> list[str]:
    """Produce a long single-source planning thread with repeated topical language."""
    return [
        (
            f"{organization} planning thread, update {index + 1}. {topic} is on the agenda again. "
            f"{details[index % len(details)]} We used this meeting to collect questions about {subject}; "
            f"the operations sheet will be circulated separately when the responsible team signs it. "
            f"The whiteboard currently records comments from the reception and facilities teams. I copied those "
            f"comments into this thread so the same discussion does not have to be repeated at the next meeting. "
            f"The next revision should shorten the introduction and keep the contact list beside the planning calendar."
        )
        for index in range(28)
    ]


def _authored_records() -> list[tuple[str, str, list[list[str]], tuple[int, ...], str]]:
    """Return questions, source notes, evidence ordinals, and evaluator answers."""
    return [
        (
            "paraphrase",
            "How does a visitor who cannot climb stairs reach the upstairs collection at Bellhaven Museum?",
            [
                [
                    "Bellhaven Museum visit note: Mum uses a wheelchair. The upper-floor ceramics are reached by "
                    "the freight lift beside the conservation studio. The attendant unlocks that lift on request; "
                    "the public elevator only serves the basement and ground level. We should ask at reception "
                    "when we arrive, before heading into the galleries. The curator confirmed this route on the telephone."
                ],
                [
                    "Bellhaven Museum access leaflet proof: the main staircase will receive new handrails. "
                    "The map shows the basement cloakroom, entrance ramp, and public elevator. The diagram is intended "
                    "for the entrance lobby; the upper-floor exhibition route is described in the separate visitor note."
                ],
            ],
            (0,),
            "Ask the attendant to unlock the freight lift beside the conservation studio.",
        ),
        (
            "paraphrase",
            "At Cinder Cafe, what hot drink did I settle on after cow's milk kept making me ill?",
            [
                [
                    "Cinder Cafe diary: another latte left me with stomach cramps, so I spoke to Petra about "
                    "switching away from dairy. I tried plain espresso but found it too bitter. The oat flat white "
                    "was smooth and caused no stomach trouble. That is my usual order from now on, with no syrup. "
                    "Petra wrote it on my loyalty card so I do not have to explain the change each visit."
                ],
                [
                    "Cinder Cafe tasting invitation: the staff are comparing a chilled coconut cocoa, almond "
                    "iced coffee, and decaffeinated mocha. I said the cold drinks sounded interesting for summer "
                    "but did not place an order. This is the supplier's sample menu, and the team has not decided "
                    "which of these drinks will be sold at the counter."
                ],
            ],
            (0,),
            "An oat flat white without syrup.",
        ),
        (
            "crowding",
            "For Arden Depot's October restart, which carrier takes refrigerated parcels and when does that collection leave?",
            [
                _crowding_notes(
                    "Arden Depot",
                    "The October restart, the carrier for refrigerated parcels, and the collection departure",
                    "the refrigerated parcel carrier and the October collection",
                    (
                        "The brochure still needs a diagram of the loading bays.",
                        "The draft checklist should use larger print for drivers reading it in the yard.",
                        "The staff training slides need an example of a completed consignment label.",
                        "The reception team wants one clear heading for the restart paperwork.",
                    ),
                ),
                [
                    "Arden Depot October restart contract signed: Frostway Logistics has accepted the chilled "
                    "shipment route. That firm handles the temperature-controlled consignments from reopening "
                    "day onward. Bram will retain the signed service agreement at the dispatch desk; vehicle "
                    "checks and insulated crate returns stay with our depot team."
                ],
                [
                    "Arden Depot October loading schedule: the refrigerated parcel run pulls out at 06:20 each "
                    "weekday. Packages must be sealed and on the marked pallet fifteen minutes earlier. The "
                    "dispatch supervisor signed this timetable for the restart and posted a copy beside the yard clock."
                ],
            ],
            (1, 2),
            "Frostway Logistics; departure at 06:20 each weekday.",
        ),
        (
            "crowding",
            "For Cedar Campus's winter shuttle, where is the accessible pickup and who covers the early driving shift?",
            [
                _crowding_notes(
                    "Cedar Campus",
                    "The winter shuttle, the accessible pickup, and early driving shift coverage",
                    "the winter shuttle pickup and the early driver",
                    (
                        "We checked the colors proposed for the passenger information posters.",
                        "The printer needs a final paper size before producing the timetable sleeves.",
                        "The communications office wants the announcement shortened to one page.",
                        "The route diagram should have space for contact details below its legend.",
                    ),
                ),
                [
                    "Cedar Campus transport office confirmation for the winter service: passengers needing "
                    "level boarding should wait beside the greenhouse gate. A lowered curb and sheltered bench "
                    "are available there. The sports pavilion stop is for the summer loop and is closed throughout "
                    "this winter service period. Facilities has completed the boarding-area inspection."
                ],
                [
                    "Cedar Campus winter minibus staff rota approved: Anika Rao takes the opening duty, from "
                    "05:45 until the morning handover. Caleb follows later in the day. This rota applies to the "
                    "winter shuttle rather than the separate evening security vehicle. Anika has acknowledged "
                    "the assignment and collected the keys from the residence office."
                ],
            ],
            (1, 2),
            "Beside the greenhouse gate; Anika Rao covers the opening duty.",
        ),
        (
            "scope_multi_source",
            "For the October Estuary Aquarium night opening, what are the first entry time and minimum visitor age?",
            [
                [
                    "Estuary Aquarium October night opening ticket notice: the first admission slot begins at "
                    "19:40. Guests should assemble by the ticket booth ten minutes beforehand. This is the October "
                    "evening event; the daytime schedule printed on the standard visitor leaflet is unchanged."
                ],
                [
                    "Estuary Aquarium safeguarding approval for the October night opening: every visitor must "
                    "be at least fourteen years old, including anyone accompanied by an adult. Clara has added "
                    "the condition to the booking form. The rule applies to the evening event only."
                ],
                [
                    "Estuary Aquarium September night opening archive: first entry was at 18:50, and the "
                    "minimum visitor age was twelve. Hana filed the closed-event paperwork after the last guest "
                    "left. This archive is retained for reconciliation of the September ticket receipts."
                ],
                [
                    "Estuary Aquarium October daytime visitor leaflet: first entry is 09:00 and children of "
                    "all ages may attend with a responsible adult. The leaflet describes ordinary public hours. "
                    "A separate booking is needed for any night opening."
                ],
                [
                    "Estuary Aquarium November night opening draft: the planning team proposed 19:10 first "
                    "entry and a minimum age of sixteen. These November proposals await approval; they are "
                    "not the ticket conditions for the October event."
                ],
            ],
            (0, 1),
            "First entry is 19:40; visitors must be at least fourteen.",
        ),
        (
            "scope_multi_source",
            "At Larch Sound's Sunday children's workshop, which room, instructor, and start time are booked?",
            [
                [
                    "Larch Sound room allocation, Sunday children's workshop: use the Willow Room. Its low "
                    "tables and washable floor are suitable for the session. The river stage remains reserved "
                    "for the afternoon concert, so the workshop supplies should be carried indoors."
                ],
                [
                    "Larch Sound teaching agreement: Yara Mensah has signed to lead the Sunday children's "
                    "workshop. Nell is handling ticket questions and will not teach that class. Yara requested "
                    "a box of chalk and a portable speaker for the movement exercises."
                ],
                [
                    "Larch Sound published Sunday programme: the children's workshop starts at 10:35. Parents "
                    "can sign children in from 10:20. The session ends before the lunch performance, leaving "
                    "volunteers enough time to wipe down the activity tables."
                ],
                [
                    "Larch Sound Saturday children's workshop booking: the Oak Room, instructor Stefan Cole, "
                    "start time 10:15. These arrangements are for Saturday. The Sunday workshop has its own "
                    "room allocation, teaching agreement, and published programme."
                ],
                [
                    "Larch Sound Sunday adult workshop booking: the Birch Room, instructor Ronan Ellis, start "
                    "time 10:35. This is the adult session. Families booking the children's workshop should use "
                    "the separate registration form because the materials and supervision differ."
                ],
            ],
            (0, 1, 2),
            "Willow Room; Yara Mensah; 10:35.",
        ),
        (
            "capacity",
            "For the Northline Bistro Tuesday tasting, give the bread supplier, dessert, drinks host, and seating capacity.",
            [
                [
                    "Northline Bistro Tuesday tasting purchase order: Hearth Finch Bakery will supply the "
                    "bread. The order is for small rye rolls, delivered before kitchen preparation begins. Theo "
                    "has checked the invoice address and confirmed the allergy declaration with the bakery."
                ],
                [
                    "Northline Bistro Tuesday tasting dessert card approved: poached pear with ginger cream. "
                    "Amara tested the portions and chose a shallow bowl for service. The kitchen will prepare "
                    "the pears in advance and add the cream immediately before the plates leave the pass."
                ],
                [
                    "Northline Bistro Tuesday tasting drinks assignment: Lena Ortiz will host the pairings. "
                    "She has agreed to introduce each pour and answer guests' questions between courses. Oscar "
                    "will handle glass washing and stock movement behind the bar."
                ],
                [
                    "Northline Bistro Tuesday tasting room plan: the seating capacity is twenty-eight guests. "
                    "The layout leaves a clear service aisle and space for coats beside the entrance. Imogen "
                    "has entered that number in the booking system and stopped automatic overbooking."
                ],
            ],
            (0, 1, 2, 3),
            "Hearth Finch Bakery; poached pear with ginger cream; Lena Ortiz; twenty-eight guests.",
        ),
        (
            "abstention",
            "What is the six-digit keypad code for the locked map cabinet at Quarry Library?",
            [
                [
                    "Quarry Library local history room note: the map cabinet must remain locked between "
                    "supervised visits. Beatrice has scheduled a conservator to inspect the drawers next month. "
                    "Readers should leave bags at the desk and use the foam supports when viewing fragile sheets."
                ],
                [
                    "Quarry Library facilities request: the keypad beside the map cabinet needs a replacement "
                    "plastic cover. Kian photographed the cracked edge for the maintenance order. The technician "
                    "will arrange access with the building manager before carrying out the repair."
                ],
            ],
            (),
            None,
        ),
    ]


def controlled_cases() -> list[tuple[EvaluationCase, GoldRecord]]:
    """Return eight fixed histories and separate evaluator records without retrieval."""
    result = []
    for case_index, (ability, question, notes, evidence, answer) in enumerate(_authored_records()):
        case_id = f"ns-{case_index + 1:02d}"
        rng = random.Random(SEED + case_index)
        sources = [_node(case_id, ordinal, texts) for ordinal, texts in enumerate(notes)]
        evidence_ids = tuple(sources[ordinal].node_id for ordinal in evidence)
        evidence_turns = tuple((node_id, "t00000") for node_id in evidence_ids)
        for ordinal in range(len(sources), SOURCE_COUNT):
            sources.append(_node(case_id, ordinal, _background(case_index, ordinal, rng)))
        rng.shuffle(sources)
        case = EvaluationCase(case_id, question, "2026-09-11", tuple(sources))
        gold = GoldRecord(
            case_id,
            answer,
            ability,
            evidence_ids,
            evidence_turns,
            {source.node_id: source.node_id for source in sources},
        )
        result.append((case, gold))
    return result


def controlled_manifest() -> dict[str, Any]:
    """Describe the frozen corpus and evaluator records with deterministic hashes."""
    cases = []
    for case, gold in controlled_cases():
        encoded = json.dumps(asdict(case), sort_keys=True, separators=(",", ":")).encode()
        labels = json.dumps(asdict(gold), sort_keys=True, separators=(",", ":")).encode()
        cases.append(
            {
                "case_id": case.case_id,
                "ability": gold.ability,
                "source_count": len(case.sources),
                "turn_count": sum(len(source.turns) for source in case.sources),
                "word_count": sum(
                    len(turn.text.split()) for source in case.sources for turn in source.turns
                ),
                "required_node_count": len(gold.evidence_node_ids),
                "case_sha256": hashlib.sha256(encoded).hexdigest(),
                "gold_sha256": hashlib.sha256(labels).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "origin": "controlled synthetic development histories; authored before retrieval",
        "seed": SEED,
        "source_count_per_history": SOURCE_COUNT,
        "isolation": "disjoint node IDs and exact source text; shared generation style is not population independence",
        "scope": "retrieval and direct seed admission only; no answer quality or benchmark generalization claim",
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(controlled_manifest(), indent=2))
