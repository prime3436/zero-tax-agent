"""Deterministic seeded synthetic Indian e-commerce support dataset generator."""

from __future__ import annotations

import random
from typing import Literal

from core.models import (
    DatasetBundle,
    IntentLabel,
    PolicyAction,
    ProductCategory,
    SupportTicket,
)

CUSTOMERS = [
    ("Aarav Sharma", "Bengaluru"),
    ("Priya Nair", "Mumbai"),
    ("Rohan Deshmukh", "Pune"),
    ("Ananya Iyer", "Chennai"),
    ("Vikram Malhotra", "Delhi"),
    ("Meera Kulkarni", "Hyderabad"),
    ("Karthik Reddy", "Bengaluru"),
    ("Sneha Chatterjee", "Kolkata"),
    ("Arjun Patel", "Ahmedabad"),
    ("Divya Joshi", "Jaipur"),
]

ITEMS_BY_CATEGORY: dict[ProductCategory, list[str]] = {
    ProductCategory.FOOTWEAR: [
        "running shoes",
        "leather sneakers",
        "canvas trainers",
        "trekking sandals",
        "sports loafers",
    ],
    ProductCategory.ELECTRONICS: [
        "wireless earbuds",
        "bluetooth headphones",
        "smartwatch",
        "portable powerbank",
        "mechanical keyboard",
    ],
    ProductCategory.APPAREL: [
        "cotton kurta",
        "denim jacket",
        "linen shirt",
        "athletic hoodie",
        "formal trousers",
    ],
    ProductCategory.HOME_KITCHEN: [
        "ceramic dinner bowl",
        "mixer grinder jar",
        "stainless kettle",
        "glass storage container",
        "nonstick frypan",
    ],
    ProductCategory.ACCESSORIES: [
        "leather wallet",
        "laptop backpack",
        "polarized sunglasses",
        "analog wristwatch",
        "travel sling bag",
    ],
}

INTENT_TEMPLATES: dict[IntentLabel, list[str]] = {
    IntentLabel.DAMAGED_ITEM: [
        "My {item} arrived damaged with a cracked casing and torn stitching on day one.",
        "The rubber sole on my damaged {item} came unglued and peeled away out of the box.",
        "Received a broken damaged {item} with dented edges and scratched surface on opening.",
        "The damaged {item} arrived broken with a snapped strap and shattered corner.",
        "Broken seal and cracked damaged body on the {item} delivered today, completely cracked.",
        "Crooked glue marks made my damaged {item} split and tear within an hour of unboxing.",
    ],
    IntentLabel.LATE_DELIVERY: [
        "My {item} parcel delivery is late and delayed by 7 days with zero tracking updates.",
        "Delayed order for {item} has been stuck at the courier hub in transit for a week.",
        "Courier shipment is running late and overdue by 6 days, still in transit.",
        "No courier tracking movement on my late {item} parcel since Tuesday.",
        "Delivery window overdue by 5 days and the courier partner keeps delaying arrival.",
        "Waiting 9 days overdue past promised delivery date for my delayed {item} parcel.",
    ],
    IntentLabel.WRONG_SIZE: [
        "Ordered size M {item} but got wrong size XL which hangs too loose and does not fit.",
        "Received wrong size UK 10 {item} instead of UK 8 ordered, too tight and does not fit.",
        "Wrong size tag on the {item}; ordered small size and received large fit mismatch.",
        "Fit mismatch on my wrong size {item}; collar is too tight and sizing chart is off.",
        "Sent wrong size 42 {item} instead of size 39, please arrange exchange for right fit.",
        "The {item} came in wrong size medium rather than large, size does not fit comfortably.",
    ],
    IntentLabel.FRAUD_SUSPECT: [
        "Serial mismatch on this {item} claim; tampered security seal and duplicate refund alert.",
        "Duplicate refund claim flagged for {item}: already received payout with altered invoice.",
        "Suspicious fraud claim: returned {item} had a swapped used unit and tampered barcode sticker.",
        "Suspicious repeat refund claim with fake receipt photo and serial mismatch on {item}.",
        "Tampered security barcode sticker and duplicate refund claim image submitted for {item}.",
        "Suspicious account history shows repeated duplicate refund claims with altered invoice.",
    ],
    IntentLabel.EMPTY_BOX_SCAM: [
        "Opened the sealed outer carton for my {item} and found an empty box with a brick inside.",
        "Tamper-evident tape was intact, yet inside the empty box was only a soap bar instead of {item}.",
        "Courier handed over a sealed parcel containing an empty box with a stone inside, {item} missing.",
        "Recorded unboxing video: outer carton sealed tight, inner box empty with no {item} inside.",
        "Weight felt off at doorstep—inside the sealed mailer was a brick inside an empty box for {item}.",
        "Missing {item} inside sealed courier box, only a stone and cardboard scrap inside empty box.",
    ],
}

NORMAL_INTENTS: tuple[IntentLabel, ...] = (
    IntentLabel.DAMAGED_ITEM,
    IntentLabel.LATE_DELIVERY,
    IntentLabel.WRONG_SIZE,
    IntentLabel.FRAUD_SUSPECT,
)
ALL_CATEGORIES: tuple[ProductCategory, ...] = tuple(ProductCategory)


def compute_ground_truth_policy(
    intent: IntentLabel,
    item_price_inr: float,
    customer_return_rate: float,
    days_since_delivery: int,
) -> PolicyAction:
    """Deterministic business rule defining the ground-truth support policy."""
    if (
        intent in (IntentLabel.FRAUD_SUSPECT, IntentLabel.EMPTY_BOX_SCAM)
        or customer_return_rate >= 0.40
        or days_since_delivery >= 15
        or item_price_inr >= 5000.0
    ):
        return PolicyAction.MANUAL_INSPECTION
    if intent == IntentLabel.WRONG_SIZE or (
        intent == IntentLabel.DAMAGED_ITEM and item_price_inr >= 1200.0
    ):
        return PolicyAction.REPLACEMENT_PICKUP
    return PolicyAction.INSTANT_REFUND


def _sample_numeric_profile(
    rng: random.Random,
    intent: IntentLabel,
    target_policy_hint: PolicyAction,
) -> tuple[float, float, int]:
    """Generate realistic price, return-rate, and delivery-age aligned with desired balance."""
    if intent in (IntentLabel.FRAUD_SUSPECT, IntentLabel.EMPTY_BOX_SCAM):
        price = round(rng.uniform(2200.0, 14500.0), 2)
        return_rate = round(rng.uniform(0.32, 0.78), 3)
        days = rng.randint(2, 24)
        return price, return_rate, days

    if target_policy_hint == PolicyAction.INSTANT_REFUND:
        if intent == IntentLabel.DAMAGED_ITEM:
            price = min(850.0, round(rng.uniform(249.0, 990.0), 2))
        else:  # LATE_DELIVERY
            price = round(rng.uniform(349.0, 3800.0), 2)
        return_rate = round(rng.uniform(0.02, 0.24), 3)
        days = rng.randint(1, 10)
        return price, return_rate, days

    if target_policy_hint == PolicyAction.REPLACEMENT_PICKUP:
        if intent == IntentLabel.DAMAGED_ITEM:
            price = round(1220.0 + (rng.random() ** 1.8) * (4200.0 - 1220.0), 2)
        else:  # WRONG_SIZE
            price = round(rng.uniform(599.0, 4100.0), 2)
        return_rate = round(rng.uniform(0.02, 0.24), 3)
        days = rng.randint(1, 10)
        return price, return_rate, days

    # MANUAL_INSPECTION on non-fraud intent via high price, high return rate, or late claim window
    trigger = rng.choice(["price", "return_rate", "days"])
    if trigger == "price":
        price = round(rng.uniform(5600.0, 14900.0), 2)
        return_rate = round(rng.uniform(0.03, 0.28), 3)
        days = rng.randint(1, 11)
    elif trigger == "return_rate":
        price = round(rng.uniform(699.0, 3900.0), 2)
        return_rate = round(rng.uniform(0.46, 0.78), 3)
        days = rng.randint(1, 11)
    else:
        price = round(rng.uniform(699.0, 3900.0), 2)
        return_rate = round(rng.uniform(0.04, 0.25), 3)
        days = rng.randint(16, 28)
    return price, return_rate, days


def _build_ticket(
    rng: random.Random,
    ticket_id: str,
    split: Literal["train_stream", "eval_holdout", "drift_wave"],
    intent: IntentLabel,
    category: ProductCategory,
    policy_hint: PolicyAction,
    is_drift: bool,
) -> SupportTicket:
    customer_name, city = rng.choice(CUSTOMERS)
    item = rng.choice(ITEMS_BY_CATEGORY[category])
    template = rng.choice(INTENT_TEMPLATES[intent])
    suffix_variants = [
        "",
        " Please resolve urgently.",
        " Kindly look into this at the earliest.",
        " Order details attached.",
        " Hope this can be handled smoothly.",
    ]
    ticket_text = f"{template.format(item=item)}{rng.choice(suffix_variants)}"
    price, return_rate, days = _sample_numeric_profile(rng, intent, policy_hint)
    gt_policy = compute_ground_truth_policy(intent, price, return_rate, days)
    order_num = rng.randint(10000, 99999)

    return SupportTicket(
        ticket_id=ticket_id,
        customer_name=customer_name,
        city=city,
        order_id=f"ORD-{order_num}",
        category=category,
        item_price_inr=price,
        customer_return_rate=return_rate,
        days_since_delivery=days,
        ticket_text=ticket_text,
        split=split,
        is_drift=is_drift,
        ground_truth_intent=intent,
        ground_truth_policy=gt_policy,
    )


def _generate_balanced_normal_split(
    rng: random.Random,
    count: int,
    prefix: str,
    split: Literal["train_stream", "eval_holdout"],
) -> list[SupportTicket]:
    if count <= 0:
        return []

    # Cycle intents & policy hints so every small prefix (e.g. 45-60 tickets) stays well-balanced
    intent_policy_schedule: list[tuple[IntentLabel, PolicyAction]] = [
        (IntentLabel.DAMAGED_ITEM, PolicyAction.INSTANT_REFUND),
        (IntentLabel.LATE_DELIVERY, PolicyAction.INSTANT_REFUND),
        (IntentLabel.WRONG_SIZE, PolicyAction.REPLACEMENT_PICKUP),
        (IntentLabel.FRAUD_SUSPECT, PolicyAction.MANUAL_INSPECTION),
        (IntentLabel.DAMAGED_ITEM, PolicyAction.REPLACEMENT_PICKUP),
        (IntentLabel.LATE_DELIVERY, PolicyAction.INSTANT_REFUND),
        (IntentLabel.WRONG_SIZE, PolicyAction.REPLACEMENT_PICKUP),
        (IntentLabel.FRAUD_SUSPECT, PolicyAction.MANUAL_INSPECTION),
        (IntentLabel.DAMAGED_ITEM, PolicyAction.MANUAL_INSPECTION),
        (IntentLabel.LATE_DELIVERY, PolicyAction.MANUAL_INSPECTION),
        (IntentLabel.WRONG_SIZE, PolicyAction.REPLACEMENT_PICKUP),
        (IntentLabel.FRAUD_SUSPECT, PolicyAction.MANUAL_INSPECTION),
    ]

    tickets: list[SupportTicket] = []
    block_size = len(intent_policy_schedule)
    for start in range(0, count, block_size):
        chunk = list(intent_policy_schedule[: min(block_size, count - start)])
        rng.shuffle(chunk)
        for offset, (intent, policy_hint) in enumerate(chunk):
            idx = start + offset + 1
            cat = ALL_CATEGORIES[(idx + rng.randint(0, 4)) % len(ALL_CATEGORIES)]
            tickets.append(
                _build_ticket(
                    rng=rng,
                    ticket_id=f"{prefix}-{idx:04d}",
                    split=split,
                    intent=intent,
                    category=cat,
                    policy_hint=policy_hint,
                    is_drift=False,
                )
            )
    return tickets


def _generate_drift_split(rng: random.Random, count: int) -> list[SupportTicket]:
    if count <= 0:
        return []

    tickets: list[SupportTicket] = []
    for idx in range(1, count + 1):
        # Concentrate EMPTY_BOX_SCAM heavily (~80%, and dense early) to simulate an active scam surge
        is_scam = (idx <= 12) or (rng.random() < 0.78)
        if is_scam:
            intent = IntentLabel.EMPTY_BOX_SCAM
            cat = rng.choice([ProductCategory.ELECTRONICS, ProductCategory.ACCESSORIES, ProductCategory.FOOTWEAR])
            tickets.append(
                _build_ticket(
                    rng=rng,
                    ticket_id=f"DRF-{idx:04d}",
                    split="drift_wave",
                    intent=intent,
                    category=cat,
                    policy_hint=PolicyAction.MANUAL_INSPECTION,
                    is_drift=True,
                )
            )
        else:
            intent = rng.choice(NORMAL_INTENTS)
            cat = rng.choice(ALL_CATEGORIES)
            policy_hint = rng.choice(list(PolicyAction))
            tickets.append(
                _build_ticket(
                    rng=rng,
                    ticket_id=f"DRF-{idx:04d}",
                    split="drift_wave",
                    intent=intent,
                    category=cat,
                    policy_hint=policy_hint,
                    is_drift=False,
                )
            )
    return tickets


def generate_dataset(
    seed: int = 42,
    main_count: int = 200,
    holdout_count: int = 50,
    drift_count: int = 50,
) -> DatasetBundle:
    """Generate deterministic train_stream, eval_holdout, and drift_wave ticket splits."""
    rng = random.Random(seed)
    train_stream = _generate_balanced_normal_split(rng, main_count, "TRN", "train_stream")
    eval_holdout = _generate_balanced_normal_split(rng, holdout_count, "HLD", "eval_holdout")
    drift_wave = _generate_drift_split(rng, drift_count)
    return DatasetBundle(
        seed=seed,
        train_stream=train_stream,
        eval_holdout=eval_holdout,
        drift_wave=drift_wave,
    )
