from contract_ai.common.schemas import BBox, ContractElement
from contract_ai.contracts.pipeline import ContractExtractor


def test_ordering_by_page_then_bbox():
    elems = [
        ContractElement(
            element_id="a",
            element_type="text",
            page_number=2,
            bbox=BBox(x0=10, y0=50, x1=20, y1=60),
            order_index=-1,
            text="A",
        ),
        ContractElement(
            element_id="b",
            element_type="text",
            page_number=1,
            bbox=BBox(x0=10, y0=100, x1=20, y1=110),
            order_index=-1,
            text="B",
        ),
        ContractElement(
            element_id="c",
            element_type="text",
            page_number=1,
            bbox=BBox(x0=10, y0=20, x1=20, y1=30),
            order_index=-1,
            text="C",
        ),
    ]

    ordered = ContractExtractor._assign_global_order(elems)
    assert [e.element_id for e in ordered] == ["c", "b", "a"]
    assert [e.order_index for e in ordered] == [1, 2, 3]
