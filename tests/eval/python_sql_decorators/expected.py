import functools

from myapp.db import db_query, query


@query(
    """
    SELECT
      u.id,
      u.name,
      COUNT(o.id) AS orders
    FROM users AS u
    LEFT JOIN orders AS o
      ON o.user_id = u.id
    GROUP BY
      u.id,
      u.name
    """
)
def users_with_order_counts():
    ...


@db_query
def active_users(min_age):
    """Users above an age threshold."""
    if min_age < 0:
        raise ValueError("min_age must be non-negative")
    return f"""
SELECT
  id,
  name
FROM users
WHERE
  active = 1 AND age >= {min_age}
ORDER BY
  name
"""


@query
async def recent_orders():
    return """
SELECT
  id,
  total
FROM orders
WHERE
  created_at > NOW() - INTERVAL '7' DAY
"""


@functools.cache
def not_decorated_with_sql():
    # Not an SQL decorator: left alone
    return "select id from users where active = 1"
