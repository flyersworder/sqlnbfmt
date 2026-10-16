import functools

from myapp.db import db_query, query


@query("select u.id, u.name, count(o.id) as orders from users u left join orders o on o.user_id = u.id group by u.id, u.name")
def users_with_order_counts():
    ...


@db_query
def active_users(min_age):
    """Users above an age threshold."""
    if min_age < 0:
        raise ValueError("min_age must be non-negative")
    return f"select id, name from users where active = 1 and age >= {min_age} order by name"


@query
async def recent_orders():
    return "select id, total from orders where created_at > now() - interval 7 day"


@functools.cache
def not_decorated_with_sql():
    # Not an SQL decorator: left alone
    return "select id from users where active = 1"
