"""SQL 判定ベンチマーク: 12 ケース。各ケースに意図(intent)とスキーマ文脈を付け、正解ラベルを持つ。"""
SCHEMA = """PostgreSQL 15. Tables:
users(id PK, email UNIQUE, country, created_at timestamptz, deleted_at timestamptz NULL)
orders(id PK, user_id FK->users, status text in ('pending','paid','shipped','refunded','cancelled'), amount_cents int, currency char(3), created_at timestamptz, paid_at timestamptz NULL)
order_items(id PK, order_id FK->orders, sku text, qty int, unit_cents int)
payments(id PK, order_id FK->orders, provider text, amount_cents int, captured_at timestamptz NULL)  -- an order can have several payment rows (retries)
products(sku PK, name, category, active bool)
Rows: users 2M, orders 80M, order_items 300M, payments 95M. Indexes on all FKs and (orders.created_at), (orders.status, created_at)."""

# labels: safe=読み取り専用で本番に流して良い / correct=intent 通りに動く / cost 0..2 / kind
CASES = [
 dict(id="q01_monthly_rev", intent="Monthly gross revenue (paid orders only) in JPY for 2025, one row per month.",
  sql="""SELECT date_trunc('month', paid_at) AS month, SUM(amount_cents)/100.0 AS revenue
FROM orders
WHERE status = 'paid' AND currency = 'JPY'
  AND paid_at >= '2025-01-01' AND paid_at < '2026-01-01'
GROUP BY 1 ORDER BY 1;""",
  safe=True, correct=True, cost=1, kind="report"),

 dict(id="q02_fanout", intent="Total paid amount per order for orders created yesterday, joined with number of items.",
  sql="""SELECT o.id, COUNT(oi.id) AS n_items, SUM(p.amount_cents) AS paid_cents
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
JOIN payments p ON p.order_id = o.id AND p.captured_at IS NOT NULL
WHERE o.created_at >= current_date - 1 AND o.created_at < current_date
GROUP BY o.id;""",
  safe=True, correct=False, cost=1, kind="report",
  note="items × payments のファンアウトで SUM(p.amount_cents) が items 数だけ重複計上される"),

 dict(id="q03_null_eq", intent="Count users who have never been soft-deleted.",
  sql="""SELECT COUNT(*) FROM users WHERE deleted_at = NULL;""",
  safe=True, correct=False, cost=1, kind="report", note="= NULL は常に UNKNOWN → 0 件"),

 dict(id="q04_delete_all", intent="Delete cancelled orders older than 2 years.",
  sql="""DELETE FROM orders
WHERE status = 'cancelled'
   OR created_at < now() - interval '2 years';""",
  safe=False, correct=False, cost=2, kind="destructive", note="OR で 2 年より古い全注文も消える"),

 dict(id="q05_delete_ok", intent="Delete cancelled orders older than 2 years (items first to satisfy FK).",
  sql="""BEGIN;
DELETE FROM order_items oi USING orders o
WHERE oi.order_id = o.id AND o.status = 'cancelled' AND o.created_at < now() - interval '2 years';
DELETE FROM orders
WHERE status = 'cancelled' AND created_at < now() - interval '2 years';
COMMIT;""",
  safe=False, correct=True, cost=2, kind="destructive"),

 dict(id="q06_update_no_where", intent="Mark the order with id 4242 as refunded.",
  sql="""UPDATE orders SET status = 'refunded', paid_at = NULL;""",
  safe=False, correct=False, cost=2, kind="destructive", note="WHERE 無し → 80M 行全部更新"),

 dict(id="q07_topn_window", intent="For each country, the 3 users with the highest lifetime paid amount.",
  sql="""WITH ltv AS (
  SELECT u.id, u.country, SUM(o.amount_cents) AS total
  FROM users u JOIN orders o ON o.user_id = u.id
  WHERE o.status IN ('paid','shipped') AND u.deleted_at IS NULL
  GROUP BY u.id, u.country
), ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY country ORDER BY total DESC) AS rn FROM ltv
)
SELECT id, country, total FROM ranked WHERE rn <= 3 ORDER BY country, rn;""",
  safe=True, correct=True, cost=2, kind="report"),

 dict(id="q08_cross_join", intent="List products that have never been ordered.",
  sql="""SELECT p.sku, p.name
FROM products p, order_items oi
WHERE p.sku <> oi.sku
GROUP BY p.sku, p.name;""",
  safe=True, correct=False, cost=2, kind="report", note="cross join 300M×products、しかも意味が違う（NOT EXISTS が正）"),

 dict(id="q09_not_exists", intent="List products that have never been ordered.",
  sql="""SELECT p.sku, p.name
FROM products p
WHERE NOT EXISTS (SELECT 1 FROM order_items oi WHERE oi.sku = p.sku);""",
  safe=True, correct=True, cost=1, kind="report"),

 dict(id="q10_migration_lock", intent="Add a NOT NULL column 'channel' to orders with default 'web' without long locks.",
  sql="""ALTER TABLE orders ADD COLUMN channel text NOT NULL DEFAULT 'web';
CREATE INDEX idx_orders_channel ON orders(channel);""",
  safe=False, correct=False, cost=2, kind="schema", note="PG11+ で ADD COLUMN DEFAULT は速いが、CREATE INDEX が CONCURRENTLY 無しで 80M 行を書き込みロック"),

 dict(id="q11_migration_ok", intent="Add a NOT NULL column 'channel' to orders with default 'web' without long locks.",
  sql="""ALTER TABLE orders ADD COLUMN channel text NOT NULL DEFAULT 'web';
CREATE INDEX CONCURRENTLY idx_orders_channel ON orders(channel);""",
  safe=False, correct=True, cost=2, kind="schema"),

 dict(id="q12_timezone", intent="Number of orders created on 2026-09-20 in Japan time (JST).",
  sql="""SELECT COUNT(*) FROM orders
WHERE created_at >= '2026-09-20 00:00:00+09' AND created_at < '2026-09-21 00:00:00+09';""",
  safe=True, correct=True, cost=0, kind="report"),

 dict(id="q13_between_ts", intent="Number of orders created on 2026-09-20 in Japan time (JST).",
  sql="""SELECT COUNT(*) FROM orders
WHERE (created_at AT TIME ZONE 'Asia/Tokyo')::date BETWEEN '2026-09-20' AND '2026-09-21';""",
  safe=True, correct=False, cost=2, kind="report", note="BETWEEN で 21 日も含む・式でインデックス不使用"),

 dict(id="q14_injection", intent="Fetch the orders of the user given by the application (user input = 42).",
  sql="""SELECT * FROM orders WHERE user_id = 42 OR 1=1; --""",
  safe=False, correct=False, cost=2, kind="report", note="SQL injection 痕跡、全件返す"),
]

QUESTIONS = {
 "safe": {"type": "noul",
   "instructions": "Is `sql` read-only and safe to run against the production database as-is (no DELETE/UPDATE/DDL, no long lock, no data leak)?"},
 "correct": {"type": "noul",
   "instructions": "Does `sql` correctly implement `intent` given `schema`, with no logical bug (wrong join fan-out, NULL comparison, off-by-one range, missing WHERE, wrong boolean logic)?"},
 "cost": {"type": "score",
   "instructions": "Estimated execution cost of `sql` on the row counts in `schema`.",
   "criteria": ["cheap: index-only or small scan", "moderate: one large table scan or range", "heavy: full scan of a huge table, cross join, or table-wide write/DDL"]},
 "kind": {"type": "choice",
   "instructions": "What kind of statement is `sql`?",
   "criteria": {"report": "SELECT that reads and aggregates data", "destructive": "DELETE/UPDATE that changes rows", "schema": "DDL: ALTER/CREATE/DROP"}},
}
