# Copyright © 2026 Apple Inc.

import argparse
import numbers


def coerce_positive_int(value, *, field_name: str) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return ivalue


def positive_int(value):
    try:
        ivalue = int(value) if isinstance(value, str) else value
        return coerce_positive_int(ivalue, field_name="value")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
