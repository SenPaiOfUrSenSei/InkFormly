import math

import pytest

from inkform.schema import Document, Line, StyleReference


@pytest.fixture
def document():
    # Analytic strokes are only test fixtures; production has no procedural alphabet.
    offsets = []
    previous = (0, 0)
    for i in range(200):
        position = (i * 2.0, math.sin(i / 8) * 15)
        offsets.append((position[0] - previous[0], position[1] - previous[1], int(i == 199)))
        previous = position
    reference = StyleReference(text="A test line", normalized_offsets=[(0.1, 0.1, 0)] * 20)
    return Document(source_text="A test line", seed=42, bias=0.8,
                    lines=[Line(text="A test line", offsets=offsets, seed=42, seconds=0,
                                attempts=1, primed=False, attention_complete=True)],
                    reference=reference, generation_seconds=0.1)
