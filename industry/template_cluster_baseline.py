import sqlparse


def parse_query(query):
    """
    Parse a SQL query into its components (SELECT, FROM, WHERE, GROUP BY, ORDER BY) using sqlparse.
    """
    parsed = sqlparse.parse(query)[0]
    tokens = parsed.tokens

    select_clause, from_clause, where_clause, group_by_clause, order_by_clause, distinct = [], [], [], [], [], False

    for token in tokens:
        if token.ttype is sqlparse.tokens.DML and token.value.upper() == "SELECT":
            distinct = "DISTINCT" in query.upper()
            select_clause = extract_columns(query, "SELECT", "FROM")
        elif token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "FROM":
            from_clause = extract_columns(query, "FROM", "WHERE" if "WHERE" in query.upper() else "GROUP BY")
        elif token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "WHERE":
            where_clause = extract_conditions(query)
        elif token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "GROUP BY":
            group_by_clause = extract_columns(query, "GROUP BY", "ORDER BY" if "ORDER BY" in query.upper() else None)
        elif token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "ORDER BY":
            order_by_clause = extract_columns(query, "ORDER BY", None)

    return {
        "SELECT": select_clause,
        "FROM": from_clause,
        "WHERE": where_clause,
        "GROUP BY": group_by_clause,
        "ORDER BY": order_by_clause,
        "DISTINCT": distinct,
    }


def extract_columns(query, start_clause, end_clause=None):
    """
    Extract columns between start_clause and end_clause in a query.
    """
    start_index = query.upper().find(start_clause)
    end_index = len(query) if not end_clause else query.upper().find(end_clause)
    clause = query[start_index + len(start_clause):end_index].strip()
    return [col.strip() for col in clause.split(",") if col.strip()]


def extract_conditions(query):
    """
    Extract individual conditions from the WHERE clause of a query.
    """
    if "WHERE" not in query.upper():
        return []

    start_index = query.upper().find("WHERE") + len("WHERE")
    end_index = query.upper().find("GROUP BY") if "GROUP BY" in query.upper() else (
        query.upper().find("ORDER BY") if "ORDER BY" in query.upper() else len(query)
    )
    where_clause = query[start_index:end_index].strip()

    conditions = [cond.strip() for cond in where_clause.split("AND") if cond.strip()]
    return conditions


def calculate_edit_distance_semantic(query1, query2):
    """
    Calculate the semantic edit distance between two SQL queries.
    """
    components1 = parse_query(query1)
    components2 = parse_query(query2)

    edit_distance = 0

    if components1["DISTINCT"] != components2["DISTINCT"]:
        edit_distance += 1

    select_diff = set(components1["SELECT"]).symmetric_difference(set(components2["SELECT"]))
    edit_distance += len(select_diff)

    from_diff = set(components1["FROM"]).symmetric_difference(set(components2["FROM"]))
    edit_distance += len(from_diff)

    where_diff = set(components1["WHERE"]).symmetric_difference(set(components2["WHERE"]))
    edit_distance += len(where_diff)

    group_by_diff = set(components1["GROUP BY"]).symmetric_difference(set(components2["GROUP BY"]))
    edit_distance += len(group_by_diff)

    order_by_diff = set(components1["ORDER BY"]).symmetric_difference(set(components2["ORDER BY"]))
    edit_distance += len(order_by_diff)

    return edit_distance

