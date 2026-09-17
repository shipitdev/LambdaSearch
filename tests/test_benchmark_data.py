from shared.benchmark_data import build_benchmark, generate_products


def test_generated_products_include_structured_search_fields():
    product = generate_products(1, seed=7)[0]

    assert product["product_type"]
    assert product["attributes"]
    assert product["use_cases"]
    assert product["search_text"]


def test_benchmark_is_deterministic_and_has_disjoint_query_splits():
    products = generate_products(300, seed=42)
    first = build_benchmark(products, seed=42, queries_per_cohort=3)
    second = build_benchmark(products, seed=42, queries_per_cohort=3)

    assert first == second
    train_queries = {case["query"] for case in first["train"]}
    test_queries = {case["query"] for case in first["test"]}
    assert train_queries.isdisjoint(test_queries)


def test_judgments_come_from_product_metadata_not_retrieval_order():
    products = generate_products(300, seed=42)
    benchmark = build_benchmark(products, seed=42, queries_per_cohort=2)

    case = benchmark["test"][0]
    assert case["judgments"]
    assert set(case["judgments"].values()) <= {1, 2, 3}
    assert "rank" not in case
